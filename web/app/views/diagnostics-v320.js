(function (global) {
  'use strict';
  var App = global.AIRApp;
  App.registerView('diagnostics', function () {
    var root = App.el('section');
    function render() {
      var history = App.state.eventHistory || [];
      App.clear(root);
      root.appendChild(App.pageHead('Observability', 'Diagnostics', 'AIR /events is a recent server window. This page may merge those windows into bounded browser-session history, which is not a durable audit log.'));
      if (App.state.lastError) root.appendChild(App.callout('warn', 'Latest runtime error', App.state.lastError));
      var timeline = App.el('div', 'timeline');
      if (!history.length) timeline.appendChild(App.el('div', 'empty', 'No recent AIR events have been observed in this browser session.'));
      history.slice().reverse().forEach(function (event) {
        var row = App.el('div', 'event'); var head = App.el('div', 'event-head');
        head.appendChild(App.el('strong', '', event.type || 'event'));
        var when = event.unix_ms ? new Date(Number(event.unix_ms)).toLocaleTimeString() : '';
        head.appendChild(App.el('small', '', '#' + (event.sequence == null ? '—' : event.sequence) + (when ? ' · ' + when : '')));
        row.appendChild(head);
        row.appendChild(App.el('code', '', (event.request_id == null ? '' : 'request ' + event.request_id + ' · ') + (event.detail || '')));
        timeline.appendChild(row);
      });
      var card = App.el('section', 'card padded'); card.appendChild(App.el('div', 'eyebrow', 'Browser-session event history')); card.appendChild(timeline); root.appendChild(card);
      var rawGrid = App.el('div', 'grid grid-2');
      var health = App.el('details', 'card padded'); health.appendChild(App.el('summary', '', 'Raw /health')); health.appendChild(App.el('pre', 'code', App.json(App.state.health))); rawGrid.appendChild(health);
      var model = App.el('details', 'card padded'); model.appendChild(App.el('summary', '', 'Raw /model')); model.appendChild(App.el('pre', 'code', App.json(App.state.model))); rawGrid.appendChild(model); root.appendChild(rawGrid);
    }
    document.addEventListener('air:state', render); render(); return root;
  });
}(window));
