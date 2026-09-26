(() => {
  "use strict";

  const canvas =
    document.getElementById("neural-ocean");

  if (!canvas) {
    return;
  }

  const ctx =
    canvas.getContext("2d", {
      alpha: true,
      desynchronized: true
    });

  const progressBar =
    document.getElementById(
      "scroll-progress-bar"
    );

  const reducedMotion =
    window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

  const coarsePointer =
    window.matchMedia(
      "(pointer: coarse)"
    ).matches;

  const saveData =
    navigator.connection?.saveData === true;

  const cpuCount =
    navigator.hardwareConcurrency || 4;

  const lowPower =
    saveData ||
    coarsePointer ||
    cpuCount <= 4;

  /*
   * This background is atmosphere, not UI.
   * It does not need native display resolution.
   */
  const PIXEL_BUDGET =
    lowPower
      ? 420000
      : 720000;

  const TARGET_FPS =
    lowPower
      ? 16
      : 22;

  const NODE_COUNT =
    lowPower
      ? 22
      : 32;

  const PARTICLE_COUNT =
    lowPower
      ? 15
      : 24;

  const CONNECTIONS_PER_NODE = 2;

  const nodes = [];
  const edges = [];
  const particles = [];

  let width = 1;
  let height = 1;
  let scale = 1;

  let activeScene = "surface";

  let pointerX = -99999;
  let pointerY = -99999;

  let running = true;
  let scrolling = false;

  let lastFrame = 0;
  let resizeTimer = 0;
  let scrollTimer = 0;
  let scrollQueued = false;

  let surfaceGradient = null;


  const sceneConfig = {

    surface: {
      network: 0.16,
      pulse: 0.15,
      particles: 0.25,
      currents: 0.20
    },

    shallows: {
      network: 0.27,
      pulse: 0.22,
      particles: 0.34,
      currents: 0.30
    },

    network: {
      network: 0.55,
      pulse: 0.60,
      particles: 0.27,
      currents: 0.30
    },

    deep: {
      network: 0.38,
      pulse: 0.34,
      particles: 0.20,
      currents: 0.18
    },

    signal: {
      network: 0.62,
      pulse: 0.78,
      particles: 0.18,
      currents: 0.28
    },

    horizon: {
      network: 0.20,
      pulse: 0.18,
      particles: 0.25,
      currents: 0.18
    }

  };


  class Node {

    constructor(index) {

      this.index = index;

      this.nx = Math.random();
      this.ny = Math.random();

      this.phase =
        Math.random() *
        Math.PI *
        2;

      this.phaseY =
        Math.random() *
        Math.PI *
        2;

      this.radius =
        0.8 +
        Math.random() *
        1.2;

      this.energy =
        Math.random() *
        0.15;

      this.x = 0;
      this.y = 0;

    }


    update(time, config) {

      /*
       * Nodes don't physically wander through
       * space anymore. They gently wobble around
       * fixed topology points.
       *
       * That lets us calculate connections once.
       */

      this.x =
        this.nx * width +
        Math.sin(
          time * 0.00018 +
          this.phase
        ) * 7;

      this.y =
        this.ny * height +
        Math.sin(
          time * 0.00014 +
          this.phaseY
        ) * 5;

      if (!coarsePointer) {

        const dx =
          pointerX -
          this.x;

        const dy =
          pointerY -
          this.y;

        const distanceSq =
          dx * dx +
          dy * dy;

        const influenceRadius = 155;
        const influenceSq =
          influenceRadius *
          influenceRadius;

        if (
          distanceSq <
          influenceSq
        ) {

          const influence =
            (
              1 -
              distanceSq /
              influenceSq
            ) *
            0.58;

          if (
            influence >
            this.energy
          ) {
            this.energy =
              influence;
          }

        }

      }

      if (
        Math.sin(
          time * 0.0008 +
          this.phase
        ) > 0.996 &&
        Math.random() <
          config.pulse *
          0.012
      ) {
        this.energy = 1;
      }

      this.energy *= 0.968;

    }

  }


  class Particle {

    constructor() {

      this.reset(true);

    }


    reset(initial = false) {

      this.x =
        Math.random() *
        width;

      this.y =
        initial
          ? Math.random() *
            height
          : height + 10;

      this.size =
        0.55 +
        Math.random() *
        0.9;

      this.speed =
        0.06 +
        Math.random() *
        0.16;

      this.drift =
        (
          Math.random() -
          0.5
        ) *
        0.04;

    }


    update() {

      this.y -=
        this.speed;

      this.x +=
        this.drift;

      if (
        this.y < -12 ||
        this.x < -20 ||
        this.x > width + 20
      ) {
        this.reset();
      }

    }

  }


  function chooseScale() {

    const pixels =
      width *
      height;

    return Math.max(
      0.38,
      Math.min(
        0.82,
        Math.sqrt(
          PIXEL_BUDGET /
          Math.max(
            pixels,
            1
          )
        )
      )
    );

  }


  function buildNodes() {

    nodes.length = 0;
    edges.length = 0;
    particles.length = 0;

    for (
      let i = 0;
      i < NODE_COUNT;
      i += 1
    ) {

      nodes.push(
        new Node(i)
      );

    }

    /*
     * Build a sparse nearest-neighbor graph
     * one time.
     */

    const edgeKeys =
      new Set();

    for (
      let i = 0;
      i < nodes.length;
      i += 1
    ) {

      const nearest = [];

      for (
        let j = 0;
        j < nodes.length;
        j += 1
      ) {

        if (i === j) {
          continue;
        }

        const dx =
          nodes[i].nx -
          nodes[j].nx;

        const dy =
          nodes[i].ny -
          nodes[j].ny;

        nearest.push({
          j,
          distance:
            dx * dx +
            dy * dy
        });

      }

      nearest.sort(
        (a, b) =>
          a.distance -
          b.distance
      );

      for (
        let k = 0;
        k <
        Math.min(
          CONNECTIONS_PER_NODE,
          nearest.length
        );
        k += 1
      ) {

        const j =
          nearest[k].j;

        const a =
          Math.min(i, j);

        const b =
          Math.max(i, j);

        const key =
          `${a}:${b}`;

        if (
          edgeKeys.has(key)
        ) {
          continue;
        }

        edgeKeys.add(key);

        edges.push({
          a,
          b,
          seed:
            Math.random()
        });

      }

    }

    for (
      let i = 0;
      i <
      PARTICLE_COUNT;
      i += 1
    ) {

      particles.push(
        new Particle()
      );

    }

  }


  function rebuildCanvas() {

    width =
      Math.max(
        1,
        window.innerWidth
      );

    height =
      Math.max(
        1,
        window.innerHeight
      );

    scale =
      chooseScale();

    canvas.width =
      Math.max(
        1,
        Math.round(
          width *
          scale
        )
      );

    canvas.height =
      Math.max(
        1,
        Math.round(
          height *
          scale
        )
      );

    canvas.style.width =
      `${width}px`;

    canvas.style.height =
      `${height}px`;

    ctx.setTransform(
      scale,
      0,
      0,
      scale,
      0,
      0
    );

    surfaceGradient =
      ctx.createRadialGradient(
        width * 0.5,
        0,
        0,
        width * 0.5,
        0,
        height
      );

    surfaceGradient
      .addColorStop(
        0,
        "rgba(82,205,244,0.050)"
      );

    surfaceGradient
      .addColorStop(
        0.38,
        "rgba(12,83,111,0.018)"
      );

    surfaceGradient
      .addColorStop(
        1,
        "rgba(0,0,0,0)"
      );

    buildNodes();

    render(
      performance.now()
    );

  }


  function scheduleResize() {

    clearTimeout(
      resizeTimer
    );

    resizeTimer =
      setTimeout(
        rebuildCanvas,
        180
      );

  }


  function drawAtmosphere(
    config
  ) {

    ctx.fillStyle =
      surfaceGradient;

    ctx.fillRect(
      0,
      0,
      width,
      height
    );


    /*
     * Three extremely faint current bands.
     */

    ctx.beginPath();

    for (
      let band = 0;
      band < 3;
      band += 1
    ) {

      const base =
        height *
        (
          0.24 +
          band *
          0.23
        );

      for (
        let x = -40;
        x < width + 40;
        x += 58
      ) {

        const y =
          base +
          Math.sin(
            x * 0.0042 +
            band * 1.8
          ) *
          (
            5 +
            band
          );

        if (x === -40) {
          ctx.moveTo(
            x,
            y
          );
        } else {
          ctx.lineTo(
            x,
            y
          );
        }

      }

    }

    ctx.strokeStyle =
      `rgba(
        88,
        206,
        239,
        ${
          0.012 *
          config.currents
        }
      )`;

    ctx.lineWidth = 1;
    ctx.stroke();

  }


  function drawParticles(
    config
  ) {

    ctx.beginPath();

    for (
      const p
      of particles
    ) {

      p.update();

      ctx.moveTo(
        p.x +
        p.size,
        p.y
      );

      ctx.arc(
        p.x,
        p.y,
        p.size,
        0,
        Math.PI *
        2
      );

    }

    ctx.fillStyle =
      `rgba(
        185,
        235,
        250,
        ${
          0.095 *
          config.particles
        }
      )`;

    ctx.fill();

  }


  function drawNetwork(
    time,
    config
  ) {

    for (
      const node
      of nodes
    ) {

      node.update(
        time,
        config
      );

    }


    /*
     * Quiet network: one draw call.
     */

    ctx.beginPath();

    for (
      const edge
      of edges
    ) {

      const a =
        nodes[edge.a];

      const b =
        nodes[edge.b];

      ctx.moveTo(
        a.x,
        a.y
      );

      ctx.lineTo(
        b.x,
        b.y
      );

    }

    ctx.strokeStyle =
      `rgba(
        75,
        208,
        246,
        ${
          0.175 *
          config.network
        }
      )`;

    ctx.lineWidth = 0.75;
    ctx.stroke();


    /*
     * Nodes: one draw call.
     */

    ctx.beginPath();

    for (
      const node
      of nodes
    ) {

      ctx.moveTo(
        node.x +
        node.radius,
        node.y
      );

      ctx.arc(
        node.x,
        node.y,
        node.radius,
        0,
        Math.PI *
        2
      );

    }

    ctx.fillStyle =
      `rgba(
        115,
        225,
        255,
        ${
          0.30 *
          config.network
        }
      )`;

    ctx.fill();


    /*
     * Only a handful of active edges get
     * animated traveling signals.
     */

    let pulses = 0;

    for (
      const edge
      of edges
    ) {

      if (
        pulses >= 7
      ) {
        break;
      }

      const a =
        nodes[edge.a];

      const b =
        nodes[edge.b];

      const energy =
        Math.max(
          a.energy,
          b.energy
        );

      if (
        energy < 0.48
      ) {
        continue;
      }

      const t =
        (
          time *
          0.00015 +
          edge.seed
        ) % 1;

      const x =
        a.x +
        (
          b.x -
          a.x
        ) *
        t;

      const y =
        a.y +
        (
          b.y -
          a.y
        ) *
        t;

      ctx.beginPath();

      ctx.arc(
        x,
        y,
        1.3 +
        energy *
        0.7,
        0,
        Math.PI *
        2
      );

      ctx.fillStyle =
        `rgba(
          130,
          238,
          255,
          ${
            0.30 +
            energy *
            0.35
          }
        )`;

      ctx.fill();

      pulses += 1;

    }

  }


  function render(time) {

    if (
      !surfaceGradient
    ) {
      return;
    }

    const config =
      sceneConfig[
        activeScene
      ] ||
      sceneConfig.surface;

    ctx.clearRect(
      0,
      0,
      width,
      height
    );

    drawAtmosphere(
      config
    );

    drawParticles(
      config
    );

    drawNetwork(
      time,
      config
    );

  }


  function loop(time) {

    if (!running) {
      return;
    }

    /*
     * Scrolling is already visually active.
     * Keep the last canvas frame frozen while
     * the browser moves page content.
     *
     * This is the largest scroll-jank reduction.
     */
    if (scrolling) {

      requestAnimationFrame(
        loop
      );

      return;

    }

    const interval =
      1000 /
      TARGET_FPS;

    if (
      time -
      lastFrame >=
      interval
    ) {

      lastFrame =
        time -
        (
          (
            time -
            lastFrame
          ) %
          interval
        );

      render(time);

    }

    requestAnimationFrame(
      loop
    );

  }


  function updateScroll() {

    scrollQueued = false;

    const doc =
      document.documentElement;

    const max =
      doc.scrollHeight -
      window.innerHeight;

    const progress =
      max > 0
        ? window.scrollY /
          max
        : 0;

    if (progressBar) {

      progressBar.style.transform =
        `scaleX(${progress})`;

    }

    scrolling = true;

    clearTimeout(
      scrollTimer
    );

    scrollTimer =
      setTimeout(
        () => {

          scrolling = false;

          /*
           * Render immediately after motion
           * settles rather than waiting for
           * the next scheduled animation frame.
           */

          render(
            performance.now()
          );

        },
        110
      );

  }


  function queueScroll() {

    if (scrollQueued) {
      return;
    }

    scrollQueued = true;

    requestAnimationFrame(
      updateScroll
    );

  }


  function observeScenes() {

    const scenes =
      document.querySelectorAll(
        ".scene"
      );

    const observer =
      new IntersectionObserver(
        entries => {

          let best = null;

          for (
            const entry
            of entries
          ) {

            if (
              !entry.isIntersecting
            ) {
              continue;
            }

            if (
              !best ||
              entry.intersectionRatio >
              best.intersectionRatio
            ) {
              best = entry;
            }

          }

          if (best) {

            activeScene =
              best.target.dataset.scene ||
              "surface";

          }

        },
        {
          threshold: [
            0.20,
            0.45,
            0.65
          ]
        }
      );

    scenes.forEach(
      scene =>
        observer.observe(
          scene
        )
    );

  }


  function observeReveals() {

    const elements =
      document.querySelectorAll(
        ".reveal"
      );

    if (reducedMotion) {

      elements.forEach(
        element =>
          element.classList.add(
            "visible"
          )
      );

      return;

    }

    const observer =
      new IntersectionObserver(
        entries => {

          for (
            const entry
            of entries
          ) {

            if (
              !entry.isIntersecting
            ) {
              continue;
            }

            entry.target
              .classList
              .add("visible");

            observer.unobserve(
              entry.target
            );

          }

        },
        {
          threshold: 0.08,
          rootMargin:
            "0px 0px -20px 0px"
        }
      );

    elements.forEach(
      element =>
        observer.observe(
          element
        )
    );

  }


  function bindPointer() {

    if (
      coarsePointer ||
      reducedMotion
    ) {
      return;
    }

    window.addEventListener(
      "pointermove",
      event => {

        pointerX =
          event.clientX;

        pointerY =
          event.clientY;

      },
      {
        passive: true
      }
    );

    window.addEventListener(
      "pointerleave",
      () => {

        pointerX = -99999;
        pointerY = -99999;

      }
    );

  }


  function activateCluster(
    index,
    count
  ) {

    const chunk =
      Math.ceil(
        nodes.length /
        count
      );

    const start =
      index *
      chunk;

    const end =
      Math.min(
        nodes.length,
        start +
        chunk
      );

    for (
      let i = start;
      i < end;
      i += 1
    ) {

      nodes[i].energy =
        0.85 +
        Math.random() *
        0.15;

    }

  }


  function bindRuntimeMap() {

    const runtimeNodes =
      document.querySelectorAll(
        ".runtime-node"
      );

    runtimeNodes.forEach(
      (element, index) => {

        element.addEventListener(
          "pointerenter",
          () => {

            activateCluster(
              index,
              runtimeNodes.length
            );

          }
        );

      }
    );

  }


  function visibilityChange() {

    if (
      document.hidden
    ) {

      running = false;
      return;

    }

    if (
      reducedMotion
    ) {

      render(
        performance.now()
      );

      return;

    }

    if (!running) {

      running = true;
      lastFrame = 0;

      requestAnimationFrame(
        loop
      );

    }

  }


  window.addEventListener(
    "resize",
    scheduleResize,
    {
      passive: true
    }
  );

  window.addEventListener(
    "scroll",
    queueScroll,
    {
      passive: true
    }
  );

  document.addEventListener(
    "visibilitychange",
    visibilityChange
  );


  rebuildCanvas();
  updateScroll();
  observeScenes();
  observeReveals();
  bindPointer();
  bindRuntimeMap();


  if (
    reducedMotion
  ) {

    render(0);

  } else {

    requestAnimationFrame(
      loop
    );

  }

})();
