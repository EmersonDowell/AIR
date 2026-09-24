(function (global) {
  'use strict';

  var App = global.AIRApp;
  if (!App) throw new Error('AIR core must load before Setup view');

  App.registerView('setup', function () {
    var root = App.el('section');

    root.appendChild(App.pageHead(
      'First run & launch',
      'Setup',
      'AIR 0.9.12 is served by air-server. This page gives public-safe installation and launch guidance without pretending the browser can execute privileged setup commands.'
    ));

    var steps = App.el('div', 'setup-steps');
    ['Clone', 'Dependencies', 'Build', 'Test', 'Model', 'Start AIR'].forEach(function (name, i) {
      steps.appendChild(App.el('div', 'setup-step' + (i === 5 && App.state.connected ? ' done' : (i === 0 ? ' current' : '')), name));
    });
    root.appendChild(steps);

    var statusGrid = App.el('div', 'grid grid-2');
    var current = App.el('section', 'card padded stack');
    current.appendChild(App.el('div', 'eyebrow', 'Current connection'));
    current.appendChild(App.el('h3', '', App.state.connected ? 'AIR is healthy' : 'AIR is not connected'));
    current.appendChild(App.el('p', '', App.state.connected ?
      'This application is reading state from the canonical AIR HTTP surface.' :
      'Build and start AIR in a terminal, then this page will reconnect automatically.'));
    current.appendChild(App.callout('', 'Local URL', 'The default public launcher serves AIR at http://127.0.0.1:8181.'));
    statusGrid.appendChild(current);

    var scope = App.el('section', 'card padded stack');
    scope.appendChild(App.el('div', 'eyebrow', 'Qualified public scope'));
    scope.appendChild(App.el('h3', '', 'AIR 0.9.12 · R&D release'));
    scope.appendChild(App.el('p', '', 'The qualified model path focuses on Qwen2-family GGUF, including the tested Qwen2.5 path. CPU/reference mode remains available when CUDA is unavailable.'));
    scope.appendChild(App.callout('warn', 'No fake installer controls', 'A future loopback air-setup helper may safely scan dependencies and launch AIR. It is not part of AIR 0.9.12, so this browser does not expose arbitrary shell or package-install endpoints.'));
    statusGrid.appendChild(scope);
    root.appendChild(statusGrid);

    var quick = App.el('section', 'card padded stack');
    quick.appendChild(App.el('div', 'eyebrow', 'Public quick start'));
    quick.appendChild(App.el('h3', '', 'Clone → build → test → run'));
    quick.appendChild(App.el('pre', 'code',
      'git clone https://github.com/<owner>/AIR.git\\n' +
      'cd AIR\\n' +
      './scripts/build.sh\\n' +
      'ctest --test-dir build --output-on-failure\\n' +
      './scripts/install-local.sh\\n' +
      './run-air.sh /path/to/supported-model.gguf'));
    quick.appendChild(App.el('p', '', 'Replace <owner> with the repository owner shown on GitHub. The README is the authoritative public installation guide.'));
    root.appendChild(quick);

    var deps = App.el('section', 'card padded stack');
    deps.appendChild(App.el('div', 'eyebrow', 'Build readiness'));
    deps.appendChild(App.el('h3', '', 'Dependencies'));
    var grid = App.el('div', 'grid grid-3');
    [
      ['C++20 compiler', 'Required'],
      ['CMake 3.22+', 'Required'],
      ['pkg-config', 'Required'],
      ['PCRE2 8-bit development headers', 'Required'],
      ['Boost Beast + JSON', 'Required'],
      ['CUDA Toolkit + cuBLAS', 'Optional GPU acceleration']
    ].forEach(function (dep) {
      var item = App.el('div', 'dependency-card');
      item.appendChild(App.el('strong', '', dep[0]));
      item.appendChild(App.el('span', '', dep[1]));
      grid.appendChild(item);
    });
    deps.appendChild(grid);
    deps.appendChild(App.callout('', 'AIR is the runtime', 'llama.cpp is not an AIR dependency. AIR directly owns its supported GGUF inference path.'));
    root.appendChild(deps);

    var next = App.el('section', 'card padded stack');
    next.appendChild(App.el('div', 'eyebrow', 'Setup roadmap'));
    next.appendChild(App.el('h3', '', 'Future local bootstrap helper'));
    next.appendChild(App.el('p', '', 'The planned air-setup component is intentionally separate from InferenceService. It may detect allowlisted dependencies, validate selected model paths, and start/stop the canonical air-server on loopback.'));
    var bullets = App.el('ul', 'clean-list');
    [
      'No arbitrary shell endpoint.',
      'No silent privilege escalation.',
      'No second scheduler or inference path.',
      'No recursive home-directory scan.',
      'Explicit user confirmation before privileged installation guidance.'
    ].forEach(function (line) { bullets.appendChild(App.el('li', '', line)); });
    next.appendChild(bullets);
    root.appendChild(next);

    document.addEventListener('air:state', function () {
      // View is intentionally mostly static. Connection state is refreshed by
      // route reconstruction when the browser reloads; top bar remains live.
    });

    return root;
  });
}(window));
