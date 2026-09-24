(function (global) {
  'use strict';

  var App = global.AIRApp;

  App.registerView('playground', function () {
    var root = App.el('section');
    var activeController = null;

    root.appendChild(App.pageHead(
      'Inference',
      'Playground',
      'Use AIR-native generation by default, or exercise the bounded OpenAI-shaped compatibility routes.'
    ));

    var layout = App.el('div', 'grid grid-2');
    var controls = App.el('section', 'card padded stack');
    var output = App.el('section', 'card padded stack');

    var mode = App.el('select', 'select');
    var modeLabels = [
      'Native AIR /generate',
      'Chat /v1/chat/completions',
      'Completion /v1/completions'
    ];

    modeLabels.forEach(function (label, index) {
      var option = App.el('option', '', label);
      option.value = String(index);
      mode.appendChild(option);
    });

    var prompt = App.el('textarea', 'textarea');
    prompt.placeholder = 'Enter a prompt for AIR…';

    var maxTokens = App.el('input', 'input');
    maxTokens.type = 'number';
    maxTokens.min = '1';
    maxTokens.value = '128';

    var temperature = App.el('input', 'input');
    temperature.type = 'number';
    temperature.step = '0.1';
    temperature.min = '0';
    temperature.value = '0';

    var topP = App.el('input', 'input');
    topP.type = 'number';
    topP.step = '0.05';
    topP.min = '0';
    topP.max = '1';
    topP.value = '1';

    var topK = App.el('input', 'input');
    topK.type = 'number';
    topK.min = '0';
    topK.value = '0';

    var seed = App.el('input', 'input');
    seed.type = 'number';
    seed.value = '0';

    var streaming = App.el('input');
    streaming.type = 'checkbox';
    streaming.checked = false;

    function field(label, input) {
      var wrapper = App.el('label', 'field');
      wrapper.appendChild(App.el('span', '', label));
      wrapper.appendChild(input);
      return wrapper;
    }

    controls.appendChild(field('Mode', mode));
    controls.appendChild(field('Prompt', prompt));

    var parameterGrid = App.el('div', 'grid grid-3');
    parameterGrid.appendChild(field('Max tokens', maxTokens));
    parameterGrid.appendChild(field('Temperature', temperature));
    parameterGrid.appendChild(field('Top-p', topP));
    parameterGrid.appendChild(field('Top-k', topK));
    parameterGrid.appendChild(field('Seed', seed));

    var streamField = App.el('label', 'field');
    streamField.appendChild(App.el('span', '', 'Streaming'));
    var streamLine = App.el('div');
    streamLine.appendChild(streaming);
    streamLine.appendChild(document.createTextNode(' stream response'));
    streamField.appendChild(streamLine);
    parameterGrid.appendChild(streamField);
    controls.appendChild(parameterGrid);

    controls.appendChild(App.callout(
      '',
      'Reliable default',
      'Streaming is off by default in Web 3.1 so the first request uses AIR’s ordinary JSON response path. Enable streaming when you want token-by-token output.'
    ));

    var buttons = App.el('div', 'actions');
    var run = App.el('button', 'button primary', 'Run');
    run.type = 'button';
    var cancel = App.el('button', 'button ghost', 'Cancel stream');
    cancel.type = 'button';
    cancel.disabled = true;
    buttons.appendChild(run);
    buttons.appendChild(cancel);
    controls.appendChild(buttons);

    output.appendChild(App.el('div', 'eyebrow', 'Response'));
    var status = App.el('div', 'badge', 'Idle');
    output.appendChild(status);

    var transport = App.el('div', 'footer-note', 'No request yet.');
    output.appendChild(transport);

    var responseText = App.el('pre', 'code', 'Run a request to see output.');
    output.appendChild(responseText);

    var metricsHost = App.el('div', 'stack');
    output.appendChild(metricsHost);

    var raw = App.el('details', 'details');
    var summary = App.el('summary', '', 'Raw response / stream frames');
    var rawPre = App.el('pre', 'code', '—');
    raw.appendChild(summary);
    raw.appendChild(rawPre);
    output.appendChild(raw);

    function numeric(input, fallback) {
      var value = Number(input.value);
      return isFinite(value) ? value : fallback;
    }

    function buildPayload() {
      var body = {
        max_tokens: Math.max(1, numeric(maxTokens, 128)),
        temperature: Math.max(0, numeric(temperature, 0)),
        top_p: Math.max(0, Math.min(1, numeric(topP, 1))),
        top_k: Math.max(0, numeric(topK, 0)),
        seed: numeric(seed, 0),
        stream: streaming.checked
      };

      if (mode.value === '1') {
        body.messages = [{ role: 'user', content: prompt.value }];
      } else {
        body.prompt = prompt.value;
      }

      return body;
    }

    function renderMetrics(data) {
      var metrics = data && data.metrics ? data.metrics : null;
      var cards;

      App.clear(metricsHost);
      if (!metrics) return;

      cards = App.el('div', 'grid grid-3');
      cards.appendChild(App.metricCard(
        'Total',
        App.formatMs(metrics.total_ms),
        'request ' + (metrics.request_id == null ? '—' : metrics.request_id)
      ));
      cards.appendChild(App.metricCard(
        'TTFT',
        App.formatMs(metrics.ttft_ms),
        'queue ' + App.formatMs(metrics.queue_ms)
      ));
      cards.appendChild(App.metricCard(
        'Decode tok/s',
        App.formatNumber(metrics.decode_tokens_per_second, 1),
        'prefix reused ' + App.formatNumber(metrics.prefix_reused_tokens, 0)
      ));
      metricsHost.appendChild(cards);
    }

    function extractText(path, data) {
      var choice;
      var delta;

      if (!data || typeof data !== 'object') return '';
      if (typeof data.text === 'string') return data.text;

      if (data.choices && data.choices[0]) {
        choice = data.choices[0];
        if (typeof choice.text === 'string') return choice.text;
        delta = choice.delta;
        if (delta && typeof delta.content === 'string') return delta.content;
        if (typeof delta === 'string') return delta;
        if (choice.message && typeof choice.message.content === 'string') return choice.message.content;
      }

      if (path === '/generate' && typeof data.content === 'string') return data.content;
      return '';
    }

    function setFailure(error) {
      var message;
      status.className = 'badge ' + (error.isBackpressure ? 'warn' : 'bad');
      status.textContent = error.isBackpressure ? 'AIR busy · HTTP 503' : (error.name === 'AbortError' ? 'Cancelled' : 'Request failed');

      if (error.isBackpressure) {
        message = 'AIR bounded the request because its submission queue is full. Retry after current work clears.';
      } else {
        message = error && error.message ? error.message : String(error);
      }
      responseText.textContent = message;
      transport.textContent = 'Request did not produce a usable response.';
    }

    run.addEventListener('click', function () {
      var body = buildPayload();
      var path = mode.value === '0' ? '/generate' : mode.value === '1' ? '/v1/chat/completions' : '/v1/completions';
      var accumulatedText = '';
      var rawFrames = [];

      if (!prompt.value.trim()) {
        status.className = 'badge warn';
        status.textContent = 'Prompt required';
        responseText.textContent = 'Enter a prompt before running inference.';
        return;
      }

      responseText.textContent = '';
      rawPre.textContent = '—';
      App.clear(metricsHost);
      transport.textContent = 'POST ' + path + ' · waiting for AIR';
      status.className = 'badge warn';
      status.textContent = body.stream ? 'Streaming' : 'Running';
      run.disabled = true;
      cancel.disabled = !body.stream;
      activeController = typeof AbortController === 'function' ? new AbortController() : null;

      if (body.stream) {
        App.api.stream(path, body, function (chunk) {
          var piece = extractText(path, chunk);
          rawFrames.push(chunk);
          if (rawFrames.length > 200) rawFrames.shift();
          if (piece) {
            accumulatedText += piece;
            responseText.textContent = accumulatedText;
          }
          if (chunk && chunk.metrics) renderMetrics(chunk);
          rawPre.textContent = App.json(rawFrames);
        }, activeController).then(function (streamSummary) {
          transport.textContent = 'POST ' + path + ' · ' + streamSummary.transport + ' · ' + streamSummary.data_frames + ' data frames · ' + streamSummary.bytes + ' bytes';

          if (streamSummary.parse_errors) {
            status.className = 'badge warn';
            status.textContent = 'Complete with stream parse warnings';
          } else if (!accumulatedText) {
            status.className = 'badge warn';
            status.textContent = 'Complete · no text received';
            responseText.textContent = 'AIR completed the streaming request, but the browser received no documented text chunks. Expand “Raw response / stream frames” and run the generation doctor to distinguish transport framing from runtime generation.';
          } else {
            status.className = 'badge good';
            status.textContent = 'Complete';
          }
        }).catch(setFailure).finally(function () {
          run.disabled = false;
          cancel.disabled = true;
          activeController = null;
        });
        return;
      }

      var call;
      if (path === '/generate') call = App.api.generate(body);
      else if (path === '/v1/chat/completions') call = App.api.chat(body);
      else call = App.api.complete(body);

      call.then(function (data) {
        var textValue = extractText(path, data);
        var completionTokens = App.value(data, 'usage.completion_tokens', null);

        rawPre.textContent = App.json(data);
        renderMetrics(data);
        App.state.lastGeneration = data;
        transport.textContent = 'POST ' + path + ' · JSON response';

        if (textValue) {
          responseText.textContent = textValue;
          status.className = 'badge good';
          status.textContent = 'Complete';
        } else {
          responseText.textContent = 'AIR returned a successful JSON response but no documented response text was present. Completion tokens: ' + (completionTokens == null ? 'not reported' : completionTokens) + '. Expand “Raw response / stream frames” for the exact payload.';
          status.className = 'badge warn';
          status.textContent = 'Complete · no text field';
        }
      }).catch(setFailure).finally(function () {
        run.disabled = false;
        cancel.disabled = true;
        activeController = null;
      });
    });

    cancel.addEventListener('click', function () {
      if (activeController) activeController.abort();
    });

    layout.appendChild(controls);
    layout.appendChild(output);
    root.appendChild(layout);
    return root;
  });
}(window));
