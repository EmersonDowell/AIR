(function (global) {
  'use strict';
  var App = global.AIRApp;
  App.registerView('home', function () {
    var root = App.el('section');
    function render() {
      var s = App.state;
      var r = s.runtime || {};
      var m = s.model || {};
      App.clear(root);
      root.appendChild(App.pageHead('Command Center', 'Home', 'AIR is a local inference runtime. This surface reports AIR state; it does not replace AIR state.'));
      var hero = App.el('section', 'card hero');
      hero.appendChild(App.el('div', 'eyebrow', 'AIR · ADAPTIVE INFERENCE RUNTIME'));
      hero.appendChild(App.el('h2', '', s.connected ? 'Runtime connected' : 'Runtime unavailable'));
      hero.appendChild(App.el('p', '', s.connected ? 'Generate, run Decision, and inspect the live runtime from one local application.' : 'The application shell is running, but AIR has not reported healthy runtime state yet.'));
      var badges = App.el('div', 'status-row');
      badges.appendChild(App.el('span', 'badge ' + (s.connected ? 'good' : 'bad'), s.connected ? 'AIR healthy' : 'AIR disconnected'));
      badges.appendChild(App.el('span', 'badge blue', 'Web 3.2.0'));
      badges.appendChild(App.el('span', 'badge', 'AIR 0.9.12'));
      hero.appendChild(badges);
      var quickActions = App.el('div', 'actions hero-actions');
      [
        ['Open Playground', 'playground', 'primary'],
        ['Run Decision', 'decision', ''],
        ['Inspect Runtime', 'runtime', 'ghost']
      ].forEach(function (item) {
        var button = App.el('button', 'button ' + item[2], item[0]);
        button.type = 'button';
        button.addEventListener('click', function () { App.navigate(item[1]); });
        quickActions.appendChild(button);
      });
      hero.appendChild(quickActions);
      root.appendChild(hero);

      var rd = App.el('div', 'callout r-and-d-note');
      rd.appendChild(App.el('strong', '', 'Public R&D release'));
      rd.appendChild(document.createTextNode(' AIR 0.9.12 is a qualified experimental checkpoint. The interface exposes real runtime state and explicit limitations rather than presenting research behavior as a finished commercial product.'));
      root.appendChild(rd);

      var metrics = App.el('div', 'grid grid-4');
      metrics.appendChild(App.metricCard('Backend', r.backend || m.backend || 'Not reported', 'Reported by AIR'));
      metrics.appendChild(App.metricCard('Active requests', App.formatNumber(r.active_requests, 0), 'Queued ' + App.formatNumber(r.queued_requests, 0)));
      metrics.appendChild(App.metricCard('Generated tok/s', App.formatNumber(r.aggregate_generated_tokens_per_second, 1), 'Aggregate runtime rate'));
      metrics.appendChild(App.metricCard('p95 total latency', App.formatMs(r.p95_total_ms), 'p50 ' + App.formatMs(r.p50_total_ms)));
      root.appendChild(metrics);

      var lower = App.el('div', 'grid grid-2');
      var model = App.el('section', 'card padded');
      model.appendChild(App.el('div', 'eyebrow', 'Loaded model'));
      model.appendChild(App.el('h3', '', m.id || 'No model metadata'));
      model.appendChild(App.kv('Architecture', m.architecture));
      model.appendChild(App.kv('Format', m.format));
      model.appendChild(App.kv('Context length', m.context_length == null ? null : App.formatNumber(m.context_length, 0)));
      model.appendChild(App.kv('Layers', m.layers == null ? null : App.formatNumber(m.layers, 0)));
      lower.appendChild(model);

      var planner = App.el('section', 'card padded');
      planner.appendChild(App.el('div', 'eyebrow', 'Strategy Lab'));
      planner.appendChild(App.el('h3', '', App.value(r, 'planner.strategy_id', 'No strategy reported')));
      planner.appendChild(App.kv('Mode', App.value(r, 'planner.mode', null)));
      planner.appendChild(App.kv('Objective', App.value(r, 'planner.objective', null)));
      planner.appendChild(App.kv('Selected backend', App.value(r, 'planner.selected_backend', null)));
      planner.appendChild(App.kv('Reason', App.value(r, 'planner.decision_reason', null)));
      lower.appendChild(planner);
      root.appendChild(lower);

      if (s.lastError) root.appendChild(App.callout('warn', 'Latest connection/runtime error', s.lastError));
      root.appendChild(App.el('div', 'footer-note', 'Superior MI Labs · Upper Michigan · local-first inference systems'));
    }
    document.addEventListener('air:state', render);
    render();
    return root;
  });
}(window));
