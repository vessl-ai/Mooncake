#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <async_simple/coro/SyncAwait.h>
#include <ylt/coro_rpc/coro_rpc_client.hpp>
#include <ylt/coro_rpc/coro_rpc_server.hpp>

#include "common/network.h"
#include "rpc_worker_pool.h"

namespace mooncake {
namespace {

using async_simple::coro::syncAwait;

class TestService {
   public:
    std::string Echo(int number, std::string text) {
        return text + ":" + std::to_string(number);
    }

    int Answer() { return 42; }

    // Holds its thread until Release is called.
    int Block() {
        blocked_.store(true);
        release_.wait();
        completed_.store(true);
        return 1;
    }

    int Ping() { return 0; }

    bool blocked() const { return blocked_.load(); }
    bool completed() const { return completed_.load(); }
    void Release() { release_promise_.set_value(); }

   private:
    std::atomic<bool> blocked_{false};
    std::atomic<bool> completed_{false};
    std::promise<void> release_promise_;
    std::shared_future<void> release_ = release_promise_.get_future().share();
};

class RpcWorkerPoolTest : public ::testing::Test {
   protected:
    void Start(coro_rpc::coro_rpc_server& server) {
        auto start_result = server.async_start();
        ASSERT_FALSE(start_result.hasResult());
    }

    void Connect(coro_rpc::coro_rpc_client& client) {
        auto ec = syncAwait(client.connect("127.0.0.1", std::to_string(port_)));
        ASSERT_FALSE(ec) << ec.message();
    }

    void SetUp() override {
        const auto ports = getFreeTcpPorts(1);
        ASSERT_EQ(ports.size(), 1u);
        port_ = ports.front();
    }

    int port_ = 0;
};

TEST_F(RpcWorkerPoolTest, ClientsCallPooledHandlersByMethodName) {
    TestService service;
    RpcWorkerPool pool(2);
    coro_rpc::coro_rpc_server server(/*thread_num=*/1, port_, "127.0.0.1",
                                     std::chrono::seconds(0),
                                     /*tcp_no_delay=*/true);
    pool.Register<&TestService::Echo>(server, service);
    pool.Register<&TestService::Answer>(server, service);
    ASSERT_NO_FATAL_FAILURE(Start(server));

    coro_rpc::coro_rpc_client client;
    ASSERT_NO_FATAL_FAILURE(Connect(client));

    auto echo = syncAwait(client.call<&TestService::Echo>(7, "seven"));
    ASSERT_TRUE(echo.has_value()) << echo.error().msg;
    EXPECT_EQ(echo.value(), "seven:7");

    auto answer = syncAwait(client.call<&TestService::Answer>());
    ASSERT_TRUE(answer.has_value()) << answer.error().msg;
    EXPECT_EQ(answer.value(), 42);
}

// With one io thread, both connections are read by the same thread. A
// pooled handler that never returns must still leave it free for the
// inline Ping of the other connection.
TEST_F(RpcWorkerPoolTest, BlockedPooledHandlerLeavesIoThreadFree) {
    TestService service;
    RpcWorkerPool pool(1);
    coro_rpc::coro_rpc_server server(/*thread_num=*/1, port_, "127.0.0.1",
                                     std::chrono::seconds(0),
                                     /*tcp_no_delay=*/true);
    server.register_handler<&TestService::Ping>(&service);
    pool.Register<&TestService::Block>(server, service);
    ASSERT_NO_FATAL_FAILURE(Start(server));

    coro_rpc::coro_rpc_client slow_client;
    ASSERT_NO_FATAL_FAILURE(Connect(slow_client));
    std::thread slow_caller([&] {
        auto blocked = syncAwait(slow_client.call<&TestService::Block>());
        EXPECT_TRUE(blocked.has_value()) << blocked.error().msg;
    });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!service.blocked() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(service.blocked());

    coro_rpc::coro_rpc_client ping_client;
    Connect(ping_client);
    auto ping = syncAwait(
        ping_client.call_for<&TestService::Ping>(std::chrono::seconds(2)));
    EXPECT_TRUE(ping.has_value()) << ping.error().msg;

    service.Release();
    slow_caller.join();
}

TEST_F(RpcWorkerPoolTest, StoppingTheListenerLeavesAHandlerToDrain) {
    TestService service;
    auto pool = std::make_unique<RpcWorkerPool>(1);
    coro_rpc::coro_rpc_server server(1, port_, "127.0.0.1",
                                     std::chrono::seconds(0), true);
    pool->Register<&TestService::Block>(server, service);
    ASSERT_NO_FATAL_FAILURE(Start(server));
    coro_rpc::coro_rpc_client client;
    ASSERT_NO_FATAL_FAILURE(Connect(client));
    std::thread caller([&] {
        syncAwait(
            client.call_for<&TestService::Block>(std::chrono::seconds(5)));
    });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!service.blocked() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(service.blocked());
    std::promise<void> listener_stopped;
    auto stopped = listener_stopped.get_future();
    auto drained = std::async(std::launch::async, [&] {
        server.stop();
        listener_stopped.set_value();
        pool.reset();
    });
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(20)),
              std::future_status::timeout);
    EXPECT_FALSE(service.completed());
    service.Release();
    drained.get();
    EXPECT_TRUE(service.completed());
    caller.join();
}

}  // namespace
}  // namespace mooncake
