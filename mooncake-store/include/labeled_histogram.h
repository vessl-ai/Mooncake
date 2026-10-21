#pragma once

// A Prometheus histogram whose label sets are all declared up front.
//
// ylt's histograms are not usable for a labelled histogram: the dynamic one
// repeats the bucket lines of earlier label sets when it serializes a later
// one (its buffer is never cleared), and the static one omits the labels on
// the _sum and _count lines. This class covers the small case the master
// needs: a handful of fixed label sets, lock-free observe, one HELP/TYPE
// header, and series that have seen no observation left out.

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mooncake {

template <size_t N>
class LabeledHistogram {
   public:
    using Labels = std::array<std::string, N>;

    LabeledHistogram(std::string name, std::string help,
                     std::vector<double> bounds, Labels label_names,
                     std::vector<Labels> label_sets)
        : name_(std::move(name)),
          help_(std::move(help)),
          bounds_(std::move(bounds)),
          label_names_(std::move(label_names)) {
        for (auto& labels : label_sets) {
            series_.push_back(std::make_unique<Series>(std::move(labels),
                                                       bounds_.size() + 1));
        }
    }

    // Records `value` in the series with these label values. Throws
    // std::invalid_argument for a label set that was not declared.
    void observe(const Labels& labels, double value) {
        Series& series = Find(labels);
        size_t bucket = 0;
        while (bucket < bounds_.size() && value > bounds_[bucket]) {
            ++bucket;
        }
        series.buckets[bucket].fetch_add(1, std::memory_order_relaxed);
        series.sum_micros.fetch_add(std::llround(value * 1e6),
                                    std::memory_order_relaxed);
    }

    // Observations at or below `le` (cumulative, as `_bucket{le=...}`).
    // `le` beyond the last bound, e.g. infinity, counts everything.
    int64_t count_at_or_below(const Labels& labels, double le) {
        Series& series = Find(labels);
        int64_t count = 0;
        for (size_t i = 0; i < series.buckets.size(); ++i) {
            if (i < bounds_.size() && bounds_[i] > le) {
                break;
            }
            count += series.buckets[i].load(std::memory_order_relaxed);
        }
        return count;
    }

    double sum(const Labels& labels) {
        return static_cast<double>(
                   Find(labels).sum_micros.load(std::memory_order_relaxed)) /
               1e6;
    }

    void serialize(std::string& out) {
        std::string body;
        for (const auto& series : series_) {
            std::vector<int64_t> counts(series->buckets.size());
            int64_t total = 0;
            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] = series->buckets[i].load(std::memory_order_relaxed);
                total += counts[i];
            }
            if (total == 0) {
                continue;
            }
            const std::string labels = LabelString(series->labels);
            int64_t cumulative = 0;
            for (size_t i = 0; i < counts.size(); ++i) {
                cumulative += counts[i];
                body.append(name_).append("_bucket{").append(labels);
                body.append(",le=\"")
                    .append(i < bounds_.size() ? FormatNumber(bounds_[i])
                                               : "+Inf")
                    .append("\"} ")
                    .append(std::to_string(cumulative))
                    .append("\n");
            }
            body.append(name_)
                .append("_sum{")
                .append(labels)
                .append("} ")
                .append(
                    FormatNumber(static_cast<double>(series->sum_micros.load(
                                     std::memory_order_relaxed)) /
                                 1e6))
                .append("\n");
            body.append(name_)
                .append("_count{")
                .append(labels)
                .append("} ")
                .append(std::to_string(total))
                .append("\n");
        }
        if (body.empty()) {
            return;
        }
        out.append("# HELP ").append(name_).append(" ").append(help_);
        out.append("\n# TYPE ").append(name_).append(" histogram\n");
        out.append(body);
    }

   private:
    struct Series {
        Series(Labels labels_in, size_t bucket_count)
            : labels(std::move(labels_in)), buckets(bucket_count) {}
        const Labels labels;
        std::vector<std::atomic<int64_t>> buckets;
        std::atomic<int64_t> sum_micros{0};
    };

    Series& Find(const Labels& labels) {
        for (auto& series : series_) {
            if (series->labels == labels) {
                return *series;
            }
        }
        throw std::invalid_argument("undeclared label set for " + name_);
    }

    std::string LabelString(const Labels& values) const {
        std::string out;
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) out.append(",");
            out.append(label_names_[i]).append("=\"").append(values[i]);
            out.append("\"");
        }
        return out;
    }

    static std::string FormatNumber(double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.10g", value);
        return buffer;
    }

    const std::string name_;
    const std::string help_;
    const std::vector<double> bounds_;
    const Labels label_names_;
    std::vector<std::unique_ptr<Series>> series_;
};

}  // namespace mooncake
