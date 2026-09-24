(function (global) {
  'use strict';

  var App = global.AIRApp = global.AIRApp || {};
  App.version = '3.2.0';
  App.views = {};
  App.state = {
    route: 'home',
    connected: false,
    health: null,
    model: null,
    runtime: null,
    events: [],
    metricsText: '',
    lastError: null,
    lastGeneration: null,
    eventHistory: []
  };

  App.escape = function (value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  };

  App.el = function (tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };

  App.clear = function (node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  };

  App.formatBytes = function (value) {
    var n = Number(value);
    var units = ['B', 'KB', 'MB', 'GB', 'TB'];
    var index = 0;
    if (!isFinite(n) || n < 0) return 'Not reported';
    while (n >= 1024 && index < units.length - 1) {
      n = n / 1024;
      index += 1;
    }
    return (index === 0 ? n.toFixed(0) : n.toFixed(n >= 10 ? 1 : 2)) + ' ' + units[index];
  };

  App.formatNumber = function (value, digits) {
    var n = Number(value);
    if (!isFinite(n)) return 'Not reported';
    return n.toLocaleString(undefined, { maximumFractionDigits: digits == null ? 2 : digits });
  };

  App.formatMs = function (value) {
    var n = Number(value);
    if (!isFinite(n)) return 'Not reported';
    if (n < 1000) return n.toFixed(n < 10 ? 1 : 0) + ' ms';
    return (n / 1000).toFixed(2) + ' s';
  };

  App.json = function (value) {
    try { return JSON.stringify(value, null, 2); } catch (error) { return String(value); }
  };

  App.value = function (object, path, fallback) {
    var parts = String(path).split('.');
    var current = object;
    var i;
    for (i = 0; i < parts.length; i += 1) {
      if (current == null || typeof current !== 'object' || !(parts[i] in current)) return fallback;
      current = current[parts[i]];
    }
    return current == null ? fallback : current;
  };

  App.metricCard = function (label, value, sub) {
    var card = App.el('div', 'card metric');
    card.appendChild(App.el('div', 'label', label));
    card.appendChild(App.el('div', 'value', value));
    if (sub) card.appendChild(App.el('div', 'sub', sub));
    return card;
  };

  App.kv = function (label, value) {
    var row = App.el('div', 'kv');
    row.appendChild(App.el('span', '', label));
    row.appendChild(App.el('strong', '', value == null || value === '' ? 'Not reported' : value));
    return row;
  };

  App.callout = function (kind, title, text) {
    var node = App.el('div', 'callout' + (kind ? ' ' + kind : ''));
    var strong = App.el('strong', '', title);
    strong.style.display = 'block';
    strong.style.marginBottom = '4px';
    node.appendChild(strong);
    node.appendChild(document.createTextNode(text));
    return node;
  };

  App.pageHead = function (eyebrow, title, description) {
    var wrap = App.el('div', 'page-head');
    var left = App.el('div');
    left.appendChild(App.el('div', 'eyebrow', eyebrow));
    left.appendChild(App.el('h2', '', title));
    left.appendChild(App.el('p', '', description));
    wrap.appendChild(left);
    return wrap;
  };

  App.navigate = function (route) {
    global.location.hash = String(route || 'home');
  };

  App.registerView = function (id, factory) {
    App.views[id] = factory;
  };

  App.emit = function (name) {
    var event;
    try { event = new CustomEvent(name, { detail: App.state }); }
    catch (error) { event = document.createEvent('CustomEvent'); event.initCustomEvent(name, false, false, App.state); }
    document.dispatchEvent(event);
  };

  App.setState = function (patch) {
    var key;
    for (key in patch) {
      if (Object.prototype.hasOwnProperty.call(patch, key)) App.state[key] = patch[key];
    }
    App.emit('air:state');
  };
}(window));
