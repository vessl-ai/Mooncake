#include "offload_read_guard.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mooncake {
namespace {

using namespace std::chrono_literals;
using Outcome = OffloadReadGuard::Outcome;

// Misses are forced with a Gate, never with timing. Reads expected to finish
// in time get a deadline far above any scheduling delay of a busy test host.
constexpr auto kLongDeadline = 5s;

// A read that blocks until Open(), the way a read on a slow disk does.
class Gate {
   public:
    void Open() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            open_ = true;
        }
        cv_.notify_all();
    }
    void Wait() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return open_; });
    }

   private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool open_ = false;
};

// Polls until pred() holds or 10 s pass.
template <typename Pred>
bool WaitFor(Pred pred) {
    const auto until = std::chrono::steady_clock::now() + 10s;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

tl::expected<int, ErrorCode> Value(int v) {
    return tl::expected<int, ErrorCode>(v);
}

TEST(OffloadReadGuardTest, ReturnsAReadThatFinishesInTime) {
    OffloadReadGuard guard(kLongDeadline, 1000ms, 2);
    int released = 0;
    Outcome outcome = Outcome::kShed;
    auto result = guard.Run<int>([] { return Value(7); },
                                 [&](const int&) { ++released; }, &outcome);
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 7);
    EXPECT_EQ(outcome, Outcome::kDone);
    EXPECT_EQ(released, 0);
    EXPECT_EQ(guard.deadline_miss_count(), 0u);
    EXPECT_FALSE(guard.ShouldShed(OffloadReadGuard::Clock::now()));
}

TEST(OffloadReadGuardTest, PassesAReadErrorThrough) {
    OffloadReadGuard guard(kLongDeadline, 1000ms, 2);
    Outcome outcome = Outcome::kShed;
    auto result = guard.Run<int>(
        [] {
            return tl::expected<int, ErrorCode>(
                tl::make_unexpected(ErrorCode::FILE_READ_FAIL));
        },
        [](const int&) {}, &outcome);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), ErrorCode::FILE_READ_FAIL);
    EXPECT_EQ(outcome, Outcome::kDone);
    // A failed read is not slow: no shedding.
    EXPECT_FALSE(guard.ShouldShed(OffloadReadGuard::Clock::now()));
}

TEST(OffloadReadGuardTest, TurnsAThrowingReadIntoAnError) {
    OffloadReadGuard guard(kLongDeadline, 1000ms, 1);
    auto result = guard.Run<int>(
        []() -> tl::expected<int, ErrorCode> {
            throw std::runtime_error("boom");
        },
        [](const int&) {});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), ErrorCode::INTERNAL_ERROR);
    // The worker survived: the next read works.
    auto next = guard.Run<int>([] { return Value(1); }, [](const int&) {});
    ASSERT_TRUE(next);
}

TEST(OffloadReadGuardTest, FailsALateReadAtTheDeadlineAndReleasesItLater) {
    OffloadReadGuard guard(50ms, 0ms, 2);
    Gate gate;
    std::atomic<bool> started{false};
    std::atomic<int> released{0};
    std::atomic<int> released_value{0};
    Outcome outcome = Outcome::kDone;

    const auto start = std::chrono::steady_clock::now();
    auto result = guard.Run<int>(
        [&] {
            started = true;
            gate.Wait();
            return Value(42);
        },
        [&](const int& value) {
            released_value = value;
            ++released;
        },
        &outcome);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS);
    EXPECT_EQ(outcome, Outcome::kDeadlineMiss);
    EXPECT_GE(elapsed, 50ms);
    EXPECT_EQ(guard.deadline_miss_count(), 1u);
    EXPECT_EQ(guard.late_running(), 1);
    EXPECT_EQ(released, 0);

    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));
    // A read a worker took before the deadline runs to the end and its result
    // is released; one still queued at the deadline is skipped.
    EXPECT_EQ(released, started ? 1 : 0);
    if (started) EXPECT_EQ(released_value, 42);
}

TEST(OffloadReadGuardTest, DoesNotReleaseALateReadThatFailed) {
    OffloadReadGuard guard(20ms, 0ms, 1);
    Gate gate;
    std::atomic<bool> started{false};
    int released = 0;
    auto result = guard.Run<int>(
        [&] {
            started = true;
            gate.Wait();
            return tl::expected<int, ErrorCode>(
                tl::make_unexpected(ErrorCode::FILE_READ_FAIL));
        },
        [&](const int&) { ++released; });
    ASSERT_FALSE(result);
    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));
    EXPECT_EQ(released, 0);
}

TEST(OffloadReadGuardTest, SkipsAReadThatMissedTheDeadlineInTheQueue) {
    // One thread: the first read holds it, the second waits in the queue past
    // its deadline and must never reach the disk.
    OffloadReadGuard guard(300ms, 0ms, 1);
    Gate gate;
    std::atomic<bool> first_started{false};
    std::atomic<int> second_reads{0};
    std::atomic<int> released{0};

    std::thread first([&] {
        auto r = guard.Run<int>(
            [&] {
                first_started = true;
                gate.Wait();
                return Value(1);
            },
            [&](const int&) { ++released; });
        EXPECT_FALSE(r);
    });
    ASSERT_TRUE(WaitFor([&] { return first_started.load(); }));
    // The first read is still within its deadline here (or just missed it);
    // either way the second one is queued behind it or shed.
    auto second = guard.Run<int>(
        [&] {
            ++second_reads;
            return Value(2);
        },
        [&](const int&) { ++released; });
    EXPECT_FALSE(second);
    first.join();

    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));
    EXPECT_EQ(second_reads, 0);
    EXPECT_EQ(released, 1);  // only the first read had a result to release
}

TEST(OffloadReadGuardTest, ShedsWhileALateReadRunsWithoutReading) {
    OffloadReadGuard guard(20ms, 0ms, 2);
    Gate gate;
    auto late = guard.Run<int>(
        [&] {
            gate.Wait();
            return Value(1);
        },
        [](const int&) {});
    ASSERT_FALSE(late);

    std::atomic<int> reads{0};
    Outcome outcome = Outcome::kDone;
    auto shed = guard.Run<int>(
        [&] {
            ++reads;
            return Value(2);
        },
        [](const int&) {}, &outcome);
    ASSERT_FALSE(shed);
    EXPECT_EQ(shed.error(), ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS);
    EXPECT_EQ(outcome, Outcome::kShed);
    EXPECT_EQ(reads, 0);
    EXPECT_EQ(guard.shed_count(), 1u);
    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));
}

TEST(OffloadReadGuardTest, ProbesOnceAfterTheWindowThenResumes) {
    // The deadline covers the probe the test holds below; the first miss is
    // forced by the gate.
    OffloadReadGuard guard(2000ms, 500ms, 2);
    Gate gate;
    auto late = guard.Run<int>(
        [&] {
            gate.Wait();
            return Value(1);
        },
        [](const int&) {});
    ASSERT_FALSE(late);
    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));

    // While the probe is on the disk, every other read is still shed.
    Gate probe_gate;
    std::atomic<bool> probe_started{false};
    ASSERT_TRUE(WaitFor(
        [&] { return !guard.ShouldShed(OffloadReadGuard::Clock::now()); }));
    Outcome probe_outcome = Outcome::kShed;
    std::thread prober([&] {
        auto r = guard.Run<int>(
            [&] {
                probe_started = true;
                probe_gate.Wait();
                return Value(3);
            },
            [](const int&) {}, &probe_outcome);
        EXPECT_TRUE(r);
    });
    ASSERT_TRUE(WaitFor([&] { return probe_started.load(); }));
    Outcome outcome = Outcome::kDone;
    auto during =
        guard.Run<int>([] { return Value(4); }, [](const int&) {}, &outcome);
    EXPECT_FALSE(during);
    EXPECT_EQ(outcome, Outcome::kShed);
    probe_gate.Open();
    prober.join();
    EXPECT_EQ(probe_outcome, Outcome::kDone);

    // The probe made it in time: reads go to the disk again, concurrently.
    EXPECT_FALSE(guard.ShouldShed(OffloadReadGuard::Clock::now()));
    auto after =
        guard.Run<int>([] { return Value(5); }, [](const int&) {}, &outcome);
    ASSERT_TRUE(after);
    EXPECT_EQ(outcome, Outcome::kDone);
}

TEST(OffloadReadGuardTest, KeepsSheddingForTheShedWindowAfterAMiss) {
    OffloadReadGuard guard(20ms, 2000ms, 2);
    Gate gate;
    auto late = guard.Run<int>(
        [&] {
            gate.Wait();
            return Value(1);
        },
        [](const int&) {});
    ASSERT_FALSE(late);
    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));

    // The late read is done, but the window from the miss still holds.
    std::atomic<int> reads{0};
    Outcome outcome = Outcome::kDone;
    auto shed = guard.Run<int>(
        [&] {
            ++reads;
            return Value(2);
        },
        [](const int&) {}, &outcome);
    ASSERT_FALSE(shed);
    EXPECT_EQ(outcome, Outcome::kShed);
    EXPECT_EQ(reads, 0);
}

TEST(OffloadReadGuardTest, AMissedProbeStartsTheWindowAgain) {
    OffloadReadGuard guard(30ms, 100ms, 2);
    Gate gate;
    auto first = guard.Run<int>(
        [&] {
            gate.Wait();
            return Value(1);
        },
        [](const int&) {});
    ASSERT_FALSE(first);
    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));
    ASSERT_TRUE(WaitFor(
        [&] { return !guard.ShouldShed(OffloadReadGuard::Clock::now()); }));

    Gate slow_again;
    Outcome outcome = Outcome::kDone;
    auto probe = guard.Run<int>(
        [&] {
            slow_again.Wait();
            return Value(2);
        },
        [](const int&) {}, &outcome);
    ASSERT_FALSE(probe);
    EXPECT_EQ(outcome, Outcome::kDeadlineMiss);
    EXPECT_EQ(guard.deadline_miss_count(), 2u);
    EXPECT_TRUE(guard.ShouldShed(OffloadReadGuard::Clock::now()));
    slow_again.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));
}

TEST(OffloadReadGuardTest, ConcurrentReadersAllFailFast) {
    OffloadReadGuard guard(30ms, 0ms, 4);
    Gate gate;
    std::atomic<int> released{0};
    std::atomic<int> failed{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            auto result = guard.Run<int>(
                [&] {
                    gate.Wait();
                    return Value(1);
                },
                [&](const int&) { ++released; });
            if (!result) ++failed;
        });
    }
    for (auto& t : readers) t.join();
    // Each reader either missed the deadline or was shed behind a miss.
    EXPECT_EQ(failed, 4);
    EXPECT_GE(guard.deadline_miss_count(), 1u);
    EXPECT_EQ(guard.deadline_miss_count() + guard.shed_count(), 4u);

    gate.Open();
    ASSERT_TRUE(WaitFor([&] { return guard.late_running() == 0; }));
    // A missed read that never left the queue is skipped, not released.
    EXPECT_LE(released, static_cast<int>(guard.deadline_miss_count()));
}

TEST(OffloadReadGuardTest, DestructorWaitsForLateReads) {
    std::atomic<int> released{0};
    Gate gate;
    std::thread opener;
    {
        OffloadReadGuard guard(1000ms, 0ms, 1);
        std::atomic<bool> started{false};
        auto result = guard.Run<int>(
            [&] {
                started = true;
                gate.Wait();
                return Value(1);
            },
            [&](const int&) { ++released; });
        ASSERT_FALSE(result);
        ASSERT_TRUE(started.load());
        // Open the gate only once the destructor is already waiting.
        opener = std::thread([&] {
            std::this_thread::sleep_for(100ms);
            gate.Open();
        });
    }
    EXPECT_EQ(released, 1);
    opener.join();
}

}  // namespace
}  // namespace mooncake
