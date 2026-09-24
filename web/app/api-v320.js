(function (global) {
  'use strict';

  var App = global.AIRApp;
  if (!App) throw new Error('AIR core must load before API client');

  function timeoutSignal(ms) {
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = null;
    if (controller) timer = setTimeout(function () { controller.abort(); }, ms);
    return { controller: controller, timer: timer };
  }

  function stopTimer(pair) {
    if (pair && pair.timer) clearTimeout(pair.timer);
  }

  function makeError(response, bodyText) {
    var detail = bodyText || response.statusText || 'Request failed';
    var error = new Error('HTTP ' + response.status + ' · ' + detail);
    error.status = response.status;
    error.detail = detail;
    error.isBackpressure = response.status === 503;
    return error;
  }

  function request(path, options, timeoutMs) {
    var pair = timeoutSignal(timeoutMs || 5000);
    var opts = options || {};
    if (pair.controller) opts.signal = pair.controller.signal;

    return fetch(path, opts).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (text) {
          throw makeError(response, text);
        });
      }
      return response;
    }).finally(function () {
      stopTimer(pair);
    });
  }

  function parseJsonResponse(response) {
    return response.text().then(function (text) {
      if (!text) return {};
      try {
        return JSON.parse(text);
      } catch (error) {
        var wrapped = new Error('AIR returned HTTP ' + response.status + ' but the body was not valid JSON: ' + error.message);
        wrapped.status = response.status;
        wrapped.detail = text;
        throw wrapped;
      }
    });
  }

  function jsonGet(path, timeoutMs) {
    return request(path, {
      headers: { Accept: 'application/json' },
      cache: 'no-store'
    }, timeoutMs).then(parseJsonResponse);
  }

  function textGet(path, timeoutMs) {
    return request(path, {
      headers: { Accept: 'text/plain' },
      cache: 'no-store'
    }, timeoutMs).then(function (response) {
      return response.text();
    });
  }

  function jsonPost(path, payload, timeoutMs) {
    return request(path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      body: JSON.stringify(payload)
    }, timeoutMs || 180000).then(parseJsonResponse);
  }

  function createSseParser(onEvent) {
    var buffer = '';
    var frameCount = 0;
    var dataFrameCount = 0;
    var doneSeen = false;
    var parseErrors = 0;
    var byteCount = 0;

    function dispatchFrame(frame) {
      var lines;
      var dataLines = [];
      var i;
      var line;
      var payload;
      var parsed;

      if (!frame || !frame.trim()) return;
      frameCount += 1;
      lines = frame.split(/\r?\n/);

      for (i = 0; i < lines.length; i += 1) {
        line = lines[i];
        if (line.indexOf('data:') === 0) {
          dataLines.push(line.slice(5).replace(/^ /, ''));
        }
      }

      if (!dataLines.length) return;
      payload = dataLines.join('\n').trim();
      if (!payload) return;

      if (payload === '[DONE]') {
        doneSeen = true;
        return;
      }

      dataFrameCount += 1;
      try {
        parsed = JSON.parse(payload);
      } catch (error) {
        parseErrors += 1;
        parsed = { raw: payload, parse_error: error.message };
      }
      onEvent(parsed);
    }

    function drain(final) {
      var separator;
      var index;
      var frame;

      while (true) {
        separator = /\r?\n\r?\n/.exec(buffer);
        if (!separator) break;
        index = separator.index;
        frame = buffer.slice(0, index);
        buffer = buffer.slice(index + separator[0].length);
        dispatchFrame(frame);
      }

      if (final && buffer.trim()) {
        dispatchFrame(buffer);
        buffer = '';
      }
    }

    return {
      push: function (text, bytes) {
        buffer += text || '';
        byteCount += Number(bytes) || 0;
        drain(false);
      },
      finish: function (text) {
        if (text) buffer += text;
        drain(true);
      },
      summary: function () {
        return {
          transport: 'sse',
          frames: frameCount,
          data_frames: dataFrameCount,
          done_seen: doneSeen,
          parse_errors: parseErrors,
          bytes: byteCount,
          residual_chars: buffer.length
        };
      }
    };
  }

  function readSse(response, onEvent) {
    if (!response.body || typeof response.body.getReader !== 'function') {
      throw new Error('Streaming response body is unavailable in this browser');
    }

    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var parser = createSseParser(onEvent);

    function pump() {
      return reader.read().then(function (result) {
        var tail;
        var text;

        if (result.done) {
          tail = decoder.decode();
          parser.finish(tail);
          return parser.summary();
        }

        text = decoder.decode(result.value, { stream: true });
        parser.push(text, result.value && result.value.byteLength);
        return pump();
      });
    }

    return pump();
  }

  App.api = {
    health: function () { return jsonGet('/health', 2500); },
    model: function () { return jsonGet('/model', 5000); },
    runtime: function () { return jsonGet('/runtime', 5000); },
    events: function () { return jsonGet('/events', 5000); },
    models: function () { return jsonGet('/v1/models', 5000); },
    metrics: function () { return textGet('/metrics', 5000); },
    generate: function (payload) { return jsonPost('/generate', payload, 180000); },
    complete: function (payload) { return jsonPost('/v1/completions', payload, 180000); },
    chat: function (payload) { return jsonPost('/v1/chat/completions', payload, 180000); },
    decide: function (payload) { return jsonPost('/decide', payload, 180000); },

    stream: function (path, payload, onEvent, externalController) {
      var controller = externalController || (typeof AbortController === 'function' ? new AbortController() : null);
      var options = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream, application/json'
        },
        body: JSON.stringify(payload)
      };

      if (controller) options.signal = controller.signal;

      return fetch(path, options).then(function (response) {
        var contentType;

        if (!response.ok) {
          return response.text().then(function (text) {
            throw makeError(response, text);
          });
        }

        contentType = String(response.headers.get('content-type') || '').toLowerCase();

        if (contentType.indexOf('application/json') !== -1) {
          return parseJsonResponse(response).then(function (data) {
            onEvent(data);
            return {
              transport: 'json-fallback',
              frames: 1,
              data_frames: 1,
              done_seen: true,
              parse_errors: 0,
              bytes: 0,
              content_type: contentType
            };
          });
        }

        return readSse(response, onEvent).then(function (summary) {
          summary.content_type = contentType || 'not reported';
          return summary;
        });
      });
    },

    _createSseParserForTests: createSseParser
  };
}(window));
