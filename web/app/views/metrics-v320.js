(function (global) {
  'use strict';
  var App = global.AIRApp;
  function parseMetrics(text) {
    var rows = [];
    String(text || '').split(/\r?\n/).forEach(function (line) {
      var match;
      if (!line || line.charAt(0) === '#') return;
      match = line.match(/^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)$/);
      if (match) rows.push({ name: match[1], value: Number(match[2]) });
    });
    return rows;
  }
  App.registerView('metrics', function () {
    var root = App.el('section');
    function render() {
      var text = App.state.metricsText || '';
      var rows = parseMetrics(text);
      App.clear(root);
      root.appendChild(App.pageHead('Prometheus surface', 'Metrics', 'Human-readable cards plus the untouched GET /metrics text. /runtime remains preferred for structured state.'));
      var cards = App.el('div', 'grid grid-4');
      rows.slice(0, 12).forEach(function (item) { cards.appendChild(App.metricCard(item.name.replace(/^air_/, '').replace(/_/g, ' '), App.formatNumber(item.value, 2), item.name)); });
      if (!rows.length) cards.appendChild(App.el('div', 'card empty', 'No Prometheus metrics received yet.'));
      root.appendChild(cards);
      var raw = App.el('section', 'card padded'); raw.appendChild(App.el('div', 'eyebrow', 'Raw /metrics')); raw.appendChild(App.el('pre', 'code', text || 'No metrics text received.')); root.appendChild(raw);
    }
    document.addEventListener('air:state', render); render(); return root;
  });
}(window));
