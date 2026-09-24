(function (global) {
  'use strict';
  var App = global.AIRApp;
  App.registerView('decision', function () {
    var root = App.el('section');
    root.appendChild(App.pageHead('Semantic Decision', 'Decision', 'Compare supplied semantic candidates with AIR. Scores are relative within the candidate set, not calibrated confidence or probability.'));
    var layout = App.el('div', 'grid grid-2');
    var form = App.el('section', 'card padded stack');
    var result = App.el('section', 'card padded stack');
    var input = App.el('textarea', 'textarea'); input.placeholder = 'Context or routing input…';
    var candidates = App.el('div', 'stack');
    var policy = App.el('select', 'select');
    ['sequence-logprob-mean', 'sequence-logprob-sum'].forEach(function (p) { var o = App.el('option', '', p); o.value = p; policy.appendChild(o); });

    function field(label, node) { var f = App.el('label', 'field'); f.appendChild(App.el('span', '', label)); f.appendChild(node); return f; }
    function addCandidate(id, text, modelText) {
      var row = App.el('div', 'grid grid-3');
      var idInput = App.el('input', 'input'); idInput.value = id || ''; idInput.placeholder = 'stable-id';
      var textInput = App.el('input', 'input'); textInput.value = text || ''; textInput.placeholder = 'Display text';
      var modelInput = App.el('input', 'input'); modelInput.value = modelText || ''; modelInput.placeholder = 'Optional model text';
      row.appendChild(field('Candidate ID', idInput)); row.appendChild(field('Display text', textInput)); row.appendChild(field('Model-facing text', modelInput));
      row.dataset.candidate = '1'; candidates.appendChild(row);
    }
    addCandidate('option-a', 'Option A', ' option a'); addCandidate('option-b', 'Option B', ' option b');
    form.appendChild(field('Input / context', input)); form.appendChild(candidates);
    var actions = App.el('div', 'actions');
    var add = App.el('button', 'button ghost', 'Add candidate'); add.type = 'button';
    var run = App.el('button', 'button primary', 'Run Decision'); run.type = 'button';
    actions.appendChild(add); actions.appendChild(run); form.appendChild(field('Scoring policy', policy)); form.appendChild(actions);
    form.appendChild(App.callout('warn', 'Score semantics', 'Decision results are candidate-set-normalized relative scores. calibrated=false and abstention_qualified=false.'));

    result.appendChild(App.el('div', 'eyebrow', 'Decision result'));
    var selected = App.el('h3', '', 'No decision yet'); result.appendChild(selected);
    var badges = App.el('div', 'status-row'); badges.appendChild(App.el('span', 'badge warn', 'calibrated=false')); badges.appendChild(App.el('span', 'badge warn', 'abstention_qualified=false')); result.appendChild(badges);
    var scoreHost = App.el('div', 'stack'); result.appendChild(scoreHost);
    var raw = App.el('pre', 'code', 'Run Decision to see the raw AIR response.'); result.appendChild(raw);

    add.addEventListener('click', function () { addCandidate('', '', ''); });
    run.addEventListener('click', function () {
      var rows = candidates.querySelectorAll('[data-candidate="1"]');
      var items = [];
      var i;
      for (i = 0; i < rows.length; i += 1) {
        var inputs = rows[i].querySelectorAll('input');
        var item = { id: inputs[0].value.trim(), text: inputs[1].value.trim() };
        if (inputs[2].value.trim()) item.model_text = inputs[2].value;
        if (item.id && item.text) items.push(item);
      }
      if (items.length < 2) { selected.textContent = 'Add at least two valid candidates.'; return; }
      run.disabled = true; selected.textContent = 'Decision running…'; App.clear(scoreHost);
      App.api.decide({ input: input.value, candidates: items, scoring_policy: policy.value, output_cardinality: 'exactly-one', determinism: 'required' })
        .then(function (data) {
          var ids = data.selected_candidate_ids || [];
          selected.textContent = ids.length ? 'Selected: ' + ids.join(', ') : 'AIR returned no selected candidate ID';
          var scores = data.scores || [];
          scores.forEach(function (score) {
            var row = App.el('div', 'score-row');
            var n = Number(score.normalized_score); var width = isFinite(n) ? Math.max(0, Math.min(100, n * 100)) : 0;
            row.appendChild(App.el('strong', '', score.candidate_id || 'candidate'));
            var track = App.el('div', 'score-track'); var fill = App.el('div', 'score-fill'); fill.style.width = width + '%'; track.appendChild(fill); row.appendChild(track);
            row.appendChild(App.el('span', '', isFinite(n) ? n.toFixed(4) : '—')); scoreHost.appendChild(row);
          });
          scoreHost.appendChild(App.kv('Applied scoring policy', data.applied_scoring_policy));
          scoreHost.appendChild(App.kv('Score semantics', data.score_semantics || 'candidate-set-normalized'));
          scoreHost.appendChild(App.kv('Candidate tokens scored', data.candidate_tokens_scored));
          scoreHost.appendChild(App.kv('Branch count', data.branch_count));
          raw.textContent = App.json(data);
        }).catch(function (error) { selected.textContent = error.isBackpressure ? 'AIR busy · bounded overload/backpressure' : 'Decision failed'; raw.textContent = error.message; })
        .finally(function () { run.disabled = false; });
    });
    layout.appendChild(form); layout.appendChild(result); root.appendChild(layout); return root;
  });
}(window));
