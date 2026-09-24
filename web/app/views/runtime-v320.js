(function (global) {
  'use strict';
  var App = global.AIRApp;
  App.registerView('runtime', function () {
    var root = App.el('section');
    function section(title, object) {
      var card = App.el('section', 'card padded');
      card.appendChild(App.el('div', 'eyebrow', title));
      if (!object || typeof object !== 'object') { card.appendChild(App.el('div', 'empty', 'Not reported by this AIR build.')); return card; }
      Object.keys(object).sort().forEach(function (key) {
        var value = object[key];
        if (value && typeof value === 'object') return;
        card.appendChild(App.kv(key.replace(/_/g, ' '), String(value)));
      });
      return card;
    }
    function render() {
      var r = App.state.runtime || {};
      App.clear(root);
      root.appendChild(App.pageHead('Runtime authority', 'Runtime', 'Structured live state from GET /runtime. Nested AIR objects stay nested; the UI does not invent a flattened parallel schema.'));
      var top = App.el('div', 'grid grid-4');
      top.appendChild(App.metricCard('Queued', App.formatNumber(r.queued_requests, 0), 'bounded submission queue'));
      top.appendChild(App.metricCard('Active', App.formatNumber(r.active_requests, 0), 'currently admitted'));
      top.appendChild(App.metricCard('Completed', App.formatNumber(r.completed_requests, 0), 'runtime total'));
      top.appendChild(App.metricCard('Overload rejects', App.formatNumber(r.rejected_overload_requests, 0), 'HTTP 503 backpressure'));
      root.appendChild(top);
      var resources = App.el('div', 'grid grid-3');
      resources.appendChild(App.metricCard('Device bytes', App.formatBytes(r.current_device_bytes), 'peak ' + App.formatBytes(r.peak_device_bytes)));
      resources.appendChild(App.metricCard('KV bytes', App.formatBytes(r.current_kv_bytes), 'peak ' + App.formatBytes(r.peak_kv_bytes)));
      resources.appendChild(App.metricCard('Prepared artifacts', App.formatBytes(r.current_prepared_artifact_bytes), 'reported by AIR'));
      root.appendChild(resources);
      var nested = App.el('div', 'grid grid-2');
      nested.appendChild(section('Planner', r.planner));
      nested.appendChild(section('Scheduler', r.scheduler));
      nested.appendChild(section('Capabilities', r.capabilities));
      nested.appendChild(section('Sequence state store', r.sequence_state_store));
      root.appendChild(nested);
      var raw = App.el('details', 'card padded'); raw.appendChild(App.el('summary', '', 'Raw /runtime JSON')); raw.appendChild(App.el('pre', 'code', App.json(r))); root.appendChild(raw);
    }
    document.addEventListener('air:state', render); render(); return root;
  });
}(window));
