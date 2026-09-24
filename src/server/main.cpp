#include "air/serving.hpp"
#include "air/version.hpp"
#include "protocol.hpp"

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <semaphore>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/socket.h>
#include <sys/time.h>
#endif

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = net::ip::tcp;

namespace {

struct Options {
    std::filesystem::path model;
    std::string host{"127.0.0.1"};
    std::uint16_t port{8181};
    air::BackendPreference backend{air::BackendPreference::automatic};
    int device{0};
    std::uint32_t workers{8};
    std::uint32_t max_connections{32};
    std::uint32_t io_timeout_seconds{30};
    air::SchedulerConfig scheduler{};
    std::filesystem::path event_log;
    std::filesystem::path web_root;
    air::ManifestConfig manifest{};
    air::ExecutionConfig execution{};
};

volatile std::sig_atomic_t stop_requested = 0;

extern "C" void request_stop(int) {
    stop_requested = 1;
}

void configure_socket_timeout(tcp::socket& socket, std::uint32_t seconds) {
#if defined(__unix__) || defined(__APPLE__)
    const timeval timeout{static_cast<time_t>(seconds), 0};
    const auto fd = socket.native_handle();
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        throw std::runtime_error("unable to configure HTTP socket timeout");
    }
#else
    (void)socket;
    (void)seconds;
#endif
}

void usage() {
    std::cout
        << "AIR Server " << air::version_string() << "\n\n"
        << "Usage:\n"
        << "  air-server -m model.gguf [options]\n\n"
        << "Options:\n"
        << "  --host ADDRESS                 default 127.0.0.1\n"
        << "  --port PORT                    default 8181\n"
        << "  --backend auto|reference|cuda  default auto\n"
        << "  --device ORDINAL               CUDA device, default 0\n"
        << "  --workers N                    HTTP worker threads, default 8\n"
        << "  --max-connections N            bounded accepted HTTP handlers, default 32\n"
        << "  --io-timeout N                 socket read/write timeout seconds, default 30\n"
        << "  --max-active N                 scheduler active-request limit\n"
        << "  --max-queued N                 bounded scheduler submission queue\n"
        << "  --token-budget N               scheduler model-token budget/cycle\n"
        << "  --prefill-quantum N             prompt tokens/request/cycle\n"
        << "  --cuda-prefill-block-linear KIND      baseline|reuse4|reuse8|dense-f32-cublas\n"
        << "  --cuda-decode-block-linear KIND       block matrices: baseline|reuse8|dense-f32-cublas\n"
        << "  --cuda-decode-output-linear KIND terminal vocab projection: baseline|reuse8\n"
        << "  --cuda-prefill-attention KIND   baseline|online-softmax\n"
        << "  --kv-page-tokens N              set both backend KV page sizes\n"
        << "  --reference-kv-page-tokens N    reference KV page size\n"
        << "  --cuda-kv-page-tokens N         CUDA KV page size\n"
        << "  --prefix-cache N               exact-prefix cache entries (capability-gated; reference today)\n"
        << "  --stream-queue N               per-request buffered stream events\n"
        << "  --manifest PATH                execution manifest path\n"
        << "  --no-manifest                  disable adaptive manifest\n"
        << "  --require-manifest             fail unless manifest validates\n"
        << "  --strategy-objective MODE      interactive|balanced|maximum-throughput|minimum-vram\n"
        << "  --strategy-horizon-tokens N    expected workload horizon for transition amortization\n"
        << "  --strategy-prepared-memory-budget-bytes N  optional prepared-state cap\n"
        << "  --event-log PATH               append runtime JSONL events\n"
        << "  --web-root PATH                override browser asset directory\n"
        << "  --version\n"
        << "  --help\n";
}

std::optional<std::uint32_t> parse_u32(std::string_view text) {
    try {
        const auto value = std::stoul(std::string(text));
        if (value > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
        return static_cast<std::uint32_t>(value);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<Options> parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        }
        if (arg == "--version") {
            std::cout << air::version_string() << '\n';
            std::exit(0);
        }
        if (arg == "-m" || arg == "--model") {
            auto value = next(); if (!value) return std::nullopt; options.model = *value;
        } else if (arg == "--host") {
            auto value = next(); if (!value) return std::nullopt; options.host = *value;
        } else if (arg == "--port") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed > 65535U) return std::nullopt;
            options.port = static_cast<std::uint16_t>(*parsed);
        } else if (arg == "--backend") {
            auto value = next(); if (!value) return std::nullopt;
            if (*value == "auto") options.backend = air::BackendPreference::automatic;
            else if (*value == "reference") options.backend = air::BackendPreference::reference;
            else if (*value == "cuda") options.backend = air::BackendPreference::cuda;
            else return std::nullopt;
        } else if (arg == "--device") {
            auto value = next(); if (!value) return std::nullopt;
            try { options.device = std::stoi(*value); } catch (...) { return std::nullopt; }
        } else if (arg == "--workers") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.workers = *parsed;
        } else if (arg == "--max-connections") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.max_connections = *parsed;
        } else if (arg == "--io-timeout") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.io_timeout_seconds = *parsed;
        } else if (arg == "--max-active") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.scheduler.max_active_requests = *parsed;
        } else if (arg == "--max-queued") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed) return std::nullopt; options.scheduler.max_queued_requests = *parsed;
        } else if (arg == "--token-budget") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.scheduler.token_budget_per_cycle = *parsed;
        } else if (arg == "--prefill-quantum") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.scheduler.prefill_quantum_tokens = *parsed;
        } else if (arg == "--cuda-prefill-block-linear") {
            auto value = next(); if (!value) return std::nullopt;
            auto tactic = air::quantized_linear_execution_kind_from_string(*value);
            if (!tactic) return std::nullopt;
            options.execution.cuda_prefill_block_linear = tactic.value();
        } else if (arg == "--cuda-decode-block-linear") {
            auto value = next(); if (!value) return std::nullopt;
            auto tactic = air::quantized_linear_execution_kind_from_string(*value);
            if (!tactic) return std::nullopt;
            options.execution.cuda_decode_block_linear = tactic.value();
        } else if (arg == "--cuda-decode-output-linear") {
            auto value = next(); if (!value) return std::nullopt;
            auto tactic = air::quantized_linear_execution_kind_from_string(*value);
            if (!tactic) return std::nullopt;
            options.execution.cuda_decode_output_linear = tactic.value();
        } else if (arg == "--cuda-prefill-attention") {
            auto value = next(); if (!value) return std::nullopt;
            auto tactic = air::attention_execution_kind_from_string(*value);
            if (!tactic) return std::nullopt;
            options.execution.cuda_prefill_attention = tactic.value();
        } else if (arg == "--kv-page-tokens") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt;
            options.scheduler.reference_kv_page_tokens = *parsed;
            options.scheduler.cuda_kv_page_tokens = *parsed;
        } else if (arg == "--reference-kv-page-tokens") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.scheduler.reference_kv_page_tokens = *parsed;
        } else if (arg == "--cuda-kv-page-tokens") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.scheduler.cuda_kv_page_tokens = *parsed;
        } else if (arg == "--prefix-cache") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed) return std::nullopt; options.scheduler.prefix_cache_entries = *parsed;
        } else if (arg == "--stream-queue") {
            auto value = next(); if (!value) return std::nullopt;
            auto parsed = parse_u32(*value); if (!parsed || *parsed == 0U) return std::nullopt; options.scheduler.stream_queue_capacity = *parsed;
        } else if (arg == "--strategy-objective") {
            auto value = next(); if (!value) return std::nullopt;
            auto objective = air::strategy_objective_from_string(*value); if (!objective) return std::nullopt;
            options.manifest.strategy.objective = objective.value();
        } else if (arg == "--strategy-horizon-tokens") {
            auto value = next(); if (!value) return std::nullopt;
            try { options.manifest.strategy.expected_horizon_tokens = std::stoull(*value); } catch (...) { return std::nullopt; }
        } else if (arg == "--strategy-prepared-memory-budget-bytes") {
            auto value = next(); if (!value) return std::nullopt;
            try { options.manifest.strategy.prepared_memory_budget_bytes = std::stoull(*value); } catch (...) { return std::nullopt; }
        } else if (arg == "--manifest") {
            auto value = next(); if (!value) return std::nullopt; options.manifest.path = *value;
        } else if (arg == "--no-manifest") {
            options.manifest.enabled = false;
        } else if (arg == "--require-manifest") {
            options.manifest.require = true;
        } else if (arg == "--event-log") {
            auto value = next(); if (!value) return std::nullopt; options.event_log = *value;
        } else if (arg == "--web-root") {
            auto value = next(); if (!value) return std::nullopt; options.web_root = *value;
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            return std::nullopt;
        }
    }
    if (options.model.empty()) return std::nullopt;
    return options;
}

std::filesystem::path discover_web_root(const Options& options, const char* argv0) {
    if (!options.web_root.empty()) return options.web_root;
#ifdef AIR_WEB_SOURCE_DIR
    const std::filesystem::path source{AIR_WEB_SOURCE_DIR};
    if (std::filesystem::exists(source / "index.html")) return source;
#endif
    std::error_code error;
    auto executable = std::filesystem::weakly_canonical(argv0, error);
    if (!error) {
        auto installed = executable.parent_path().parent_path() / "share" / "air" / "web";
        if (std::filesystem::exists(installed / "index.html")) return installed;
    }
    return {};
}

std::string mime_type(const std::filesystem::path& path) {
    const auto extension = path.extension().string();
    if (extension == ".html") return "text/html; charset=utf-8";
    if (extension == ".js") return "application/javascript; charset=utf-8";
    if (extension == ".css") return "text/css; charset=utf-8";
    if (extension == ".json") return "application/json; charset=utf-8";
    if (extension == ".svg") return "image/svg+xml";
    return "application/octet-stream";
}

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return std::nullopt;
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

void write_text(tcp::socket& socket,
                http::status status,
                std::string body,
                std::string_view content_type = "application/json; charset=utf-8") {
    http::response<http::string_body> response{status, 11};
    response.set(http::field::server, "AIR/" + air::version_string());
    response.set(http::field::content_type, content_type);
    response.set(http::field::cache_control, "no-store");
    response.keep_alive(false);
    response.body() = std::move(body);
    response.prepare_payload();
    beast::error_code error;
    http::write(socket, response, error);
}

void write_error(tcp::socket& socket, http::status status, std::string_view message) {
    write_text(socket, status, air::server::error_json(message));
}

bool write_all(tcp::socket& socket, std::string_view data) {
    beast::error_code error;
    net::write(socket, net::buffer(data.data(), data.size()), error);
    return !error;
}

void handle_stream(tcp::socket& socket,
                   air::InferenceService& service,
                   const air::server::ParsedRequest& parsed) {
    const std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n\r\n";
    if (!write_all(socket, headers)) return;

    bool first = true;
    auto result = service.generate(parsed.inference,
        [&](std::string_view delta, air::TokenId token) -> air::Status {
            const auto json = air::server::stream_chunk_json(parsed, delta, token, first);
            first = false;
            const std::string event = "data: " + json + "\n\n";
            if (!write_all(socket, event)) return air::Status::cancelled("stream client disconnected");
            return air::Status::ok();
        });
    if (!result) {
        const std::string event = "data: " + air::server::error_json(result.status().message()) + "\n\n";
        write_all(socket, event);
        write_all(socket, "data: [DONE]\n\n");
        return;
    }
    const auto finish = air::server::stream_finish_json(parsed, result.value());
    write_all(socket, "data: " + finish + "\n\n");
    write_all(socket, "data: [DONE]\n\n");
}

void handle_request(tcp::socket socket,
                    air::InferenceService& service,
                    const std::filesystem::path& web_root) {
    beast::flat_buffer buffer;
    http::request<http::string_body> request;
    beast::error_code error;
    http::read(socket, buffer, request, error);
    if (error) return;

    const std::string target(request.target());
    if (request.method() == http::verb::get && target == "/health") {
        write_text(socket, http::status::ok,
                   std::string("{\"status\":\"ok\",\"backend\":\"") + service.backend_name() + "\"}");
        return;
    }
    if (request.method() == http::verb::get && target == "/model") {
        write_text(socket, http::status::ok, air::server::model_json(service.model(), service.backend_name()));
        return;
    }
    if (request.method() == http::verb::get && target == "/runtime") {
        write_text(socket, http::status::ok, air::server::service_snapshot_json(service.snapshot()));
        return;
    }
    if (request.method() == http::verb::get && target == "/events") {
        write_text(socket, http::status::ok, air::server::events_json(service.recent_events()));
        return;
    }
    if (request.method() == http::verb::get && target == "/metrics") {
        write_text(socket, http::status::ok, air::server::metrics_text(service.snapshot()), "text/plain; version=0.0.4");
        return;
    }
    if (request.method() == http::verb::get && target == "/v1/models") {
        write_text(socket, http::status::ok, air::server::models_json(service.model()));
        return;
    }

    if (request.method() == http::verb::post && target == "/decide") {
        auto parsed = air::server::parse_decision_request(request.body());
        if (!parsed) {
            const auto status = parsed.status().code() == air::ErrorCode::unsupported
                ? http::status::not_implemented
                : http::status::bad_request;
            write_error(socket, status, parsed.status().message());
            return;
        }
        auto response = service.decide(parsed.value().decision);
        if (!response) {
            const auto status = response.status().code() == air::ErrorCode::unsupported
                ? http::status::not_implemented
                : (response.status().code() == air::ErrorCode::invalid_state
                    ? http::status::service_unavailable
                    : http::status::bad_request);
            write_error(socket, status, response.status().message());
            return;
        }
        write_text(socket, http::status::ok,
                   air::server::decision_json(response.value()));
        return;
    }

    if (request.method() == http::verb::post &&
        (target == "/generate" || target == "/v1/completions" || target == "/v1/chat/completions")) {
        auto parsed = air::server::parse_generation_request(target, request.body());
        if (!parsed) {
            const auto status = parsed.status().code() == air::ErrorCode::unsupported
                ? http::status::not_implemented
                : http::status::bad_request;
            write_error(socket, status, parsed.status().message());
            return;
        }
        if (parsed.value().stream) {
            handle_stream(socket, service, parsed.value());
            return;
        }
        auto response = service.generate(parsed.value().inference);
        if (!response) {
            const auto status = response.status().code() == air::ErrorCode::unsupported
                                    ? http::status::not_implemented
                                    : (response.status().code() == air::ErrorCode::invalid_state
                                        ? http::status::service_unavailable
                                        : http::status::bad_request);
            write_error(socket, status, response.status().message());
            return;
        }
        write_text(socket, http::status::ok,
                   air::server::completion_json(parsed.value(), response.value()));
        return;
    }

    if (request.method() == http::verb::get) {
        std::filesystem::path relative = target == "/" ? "index.html" : target.substr(1);
        if (relative.string().find("..") != std::string::npos) {
            write_error(socket, http::status::bad_request, "invalid asset path");
            return;
        }
        if (!web_root.empty()) {
            auto file = read_file(web_root / relative);
            if (file) {
                write_text(socket, http::status::ok, std::move(*file), mime_type(relative));
                return;
            }
        }
    }
    write_error(socket, http::status::not_found, "not found");
}

} // namespace

int main(int argc, char** argv) {
    auto options = parse_options(argc, argv);
    if (!options) {
        usage();
        return 2;
    }

    auto service = air::InferenceService::create(options->model,
                                                 options->backend,
                                                 options->device,
                                                 options->scheduler,
                                                 options->event_log,
                                                 options->manifest,
                                                 options->execution);
    if (!service) {
        std::cerr << "failed to start AIR: " << service.status().message() << '\n';
        return 3;
    }
    const auto web_root = discover_web_root(*options, argv[0]);

    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);

    try {
        net::io_context context;
        const auto address = net::ip::make_address(options->host);
        tcp::acceptor acceptor(context, {address, options->port});
        acceptor.non_blocking(true);
        std::counting_semaphore<> connection_slots(static_cast<std::ptrdiff_t>(options->max_connections));
        net::thread_pool workers(options->workers);
        std::cout << "AIR " << air::version_string()
                  << "\nModel:   " << service.value()->model().fingerprint().model_id
                  << "\nBackend: " << service.value()->backend_name()
                  << "\nListen:  http://" << options->host << ':' << options->port
                  << "\nWeb:     " << (web_root.empty() ? "unavailable" : web_root.string())
                  << '\n';

        while (!stop_requested) {
            if (!connection_slots.try_acquire_for(std::chrono::milliseconds(50))) continue;
            tcp::socket socket(context);
            beast::error_code accept_error;
            acceptor.accept(socket, accept_error);
            if (accept_error == net::error::would_block || accept_error == net::error::try_again) {
                connection_slots.release();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (accept_error) {
                connection_slots.release();
                if (stop_requested) break;
                throw beast::system_error(accept_error);
            }
            configure_socket_timeout(socket, options->io_timeout_seconds);
            try {
                net::post(workers, [socket = std::move(socket),
                                    service_ptr = service.value().get(),
                                    web_root,
                                    &connection_slots]() mutable {
                    try {
                        handle_request(std::move(socket), *service_ptr, web_root);
                    } catch (const std::exception& error) {
                        std::cerr << "request handler error: " << error.what() << '\n';
                    } catch (...) {
                        std::cerr << "request handler error: unknown exception\n";
                    }
                    connection_slots.release();
                });
            } catch (...) {
                connection_slots.release();
                throw;
            }
        }
        service.value()->shutdown();
        workers.join();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "server error: " << error.what() << '\n';
        return 4;
    }
}
