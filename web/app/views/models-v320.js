(function (global) {
  'use strict';
  var App = global.AIRApp;
  App.registerView('models', function () {
    var root = App.el('section');
    function render() {
      var m = App.state.model || {};
      App.clear(root);
      root.appendChild(App.pageHead('Model truth', 'Models', 'AIR owns the active model. This page reports the loaded model and qualified public support boundary without maintaining a browser model registry.'));
      var grid = App.el('div', 'grid grid-2');
      var current = App.el('section', 'card padded');
      current.appendChild(App.el('div', 'eyebrow', 'Active model'));
      current.appendChild(App.el('h3', '', m.id || 'No model reported'));
      current.appendChild(App.kv('Architecture', m.architecture));
      current.appendChild(App.kv('Format', m.format));
      current.appendChild(App.kv('Backend', m.backend));
      current.appendChild(App.kv('Layers', m.layers));
      current.appendChild(App.kv('Embedding', m.embedding));
      current.appendChild(App.kv('Context length', m.context_length));
      current.appendChild(App.kv('Vocabulary size', m.vocabulary_size));
      grid.appendChild(current);
      var support = App.el('section', 'card padded stack');
      support.appendChild(App.el('div', 'eyebrow', 'AIR 0.9.12 public scope'));
      support.appendChild(App.el('h3', '', 'Qwen2-family GGUF'));
      support.appendChild(App.el('p', '', 'The qualified public scope focuses on Qwen2-family GGUF, including the tested Qwen2.5 path. AIR validates the selected GGUF at launch.'));
      support.appendChild(App.callout('warn', 'No browser hot-swap', 'AIR 0.9.12 does not expose a runtime model-load/hot-swap endpoint. Change models by restarting the canonical server through the launcher or future air-setup helper.'));
      grid.appendChild(support);
      root.appendChild(grid);
    }
    document.addEventListener('air:state', render); render(); return root;
  });
}(window));
