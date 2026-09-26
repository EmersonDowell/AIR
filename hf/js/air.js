(() => {
  "use strict";

  const canvas =
    document.getElementById("neural-ocean");

  const ctx =
    canvas.getContext("2d", {
      alpha: true
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
   * The canvas is atmospheric rather than
   * precision UI. Rendering it below native
   * screen resolution on large displays saves
   * a lot of GPU fill work with almost no
   * visible quality loss.
   */
  const MAX_CANVAS_PIXELS =
    lowPower
      ? 1100000
      : 1700000;

  const TARGET_FPS =
    lowPower
      ? 24
      : 30;

  const SCROLL_FPS = 20;

  const MAX_NODES =
    lowPower
      ? 32
      : 48;

  const MAX_PARTICLES =
    lowPower
      ? 30
      : 48;

  const MAX_CONNECTIONS = 3;

  let width = 0;
  let height = 0;
  let renderScale = 1;

  let activeScene = "surface";

  let pointerX = -10000;
  let pointerY = -10000;

  let running = true;
  let scrolling = false;

  let lastFrameTime = 0;
  let resizeTimer = 0;
  let scrollTimer = 0;
  let scrollQueued = false;

  let backgroundGradient = null;

  const nodes = [];
  const particles = [];
  const activeEdges = [];

  let connectionCounts =
    new Uint8Array(0);


  const sceneConfig = {

    surface: {
      network: 0.16,
      pulse: 0.15,
      particle: 0.30,
      cyan: 0.62
    },

    shallows: {
      network: 0.28,
      pulse: 0.23,
      particle: 0.42,
      cyan: 0.72
    },

    network: {
      network: 0.52,
      pulse: 0.58,
      particle: 0.34,
      cyan: 0.94
    },

    deep: {
      network: 0.38,
      pulse: 0.36,
      particle: 0.25,
      cyan: 0.74
    },

    signal: {
      network: 0.58,
      pulse: 0.74,
      particle: 0.24,
      cyan: 1
    },

    horizon: {
      network: 0.20,
      pulse: 0.20,
      particle: 0.32,
      cyan: 0.67
    }

  };


  class Node {

    constructor() {
      this.reset();
    }

    reset() {

      this.x =
        Math.random() * width;

      this.y =
        Math.random() * height;

      this.vx =
        (Math.random() - 0.5) *
        0.11;

      this.vy =
        (Math.random() - 0.5) *
        0.08;

      this.radius =
        0.8 +
        Math.random() * 1.25;

      this.energy =
        Math.random() * 0.20;

      this.phase =
        Math.random() *
        Math.PI * 2;

    }

    update(time, config) {

      this.x += this.vx;
      this.y += this.vy;

      if (this.x < -40) {
        this.x = width + 40;
      }

      if (this.x > width + 40) {
        this.x = -40;
      }

      if (this.y < -40) {
        this.y = height + 40;
      }

      if (this.y > height + 40) {
        this.y = -40;
      }

      if (!coarsePointer) {

        const dx =
          pointerX - this.x;

        const dy =
          pointerY - this.y;

        const distSq =
          dx * dx + dy * dy;

        const radius = 165;
        const radiusSq =
          radius * radius;

        if (distSq < radiusSq) {

          const influence =
            (
              1 -
              distSq / radiusSq
            ) * 0.62;

          if (
            influence >
            this.energy
          ) {
            this.energy =
              influence;
          }

        }

      }

      const wave =
        Math.sin(
          time * 0.001 +
          this.phase
        );

      if (
        wave > 0.992 &&
        Math.random() <
          config.pulse * 0.018
      ) {
        this.energy = 1;
      }

      this.energy *= 0.971;

    }

  }


  class Particle {

    constructor() {
      this.reset(true);
    }

    reset(initial = false) {

      this.x =
        Math.random() * width;

      this.y =
        initial
          ? Math.random() * height
          : height + 20;

      this.size =
        0.5 +
        Math.random() * 1.15;

      this.speed =
        0.08 +
        Math.random() * 0.22;

      this.drift =
        (Math.random() - 0.5) *
        0.055;

    }

    update() {

      this.y -= this.speed;
      this.x += this.drift;

      if (
        this.y < -20 ||
        this.x < -30 ||
        this.x > width + 30
      ) {
        this.reset();
      }

    }

  }


  function chooseRenderScale() {

    const deviceScale =
      Math.min(
        window.devicePixelRatio || 1,
        1.25
      );

    const screenPixels =
      width * height;

    const budgetScale =
      Math.sqrt(
        MAX_CANVAS_PIXELS /
        Math.max(
          screenPixels,
          1
        )
      );

    return Math.max(
      0.55,
      Math.min(
        deviceScale,
        budgetScale
      )
    );

  }


  function resize() {

    width =
      window.innerWidth;

    height =
      window.innerHeight;

    renderScale =
      chooseRenderScale();

    canvas.width =
      Math.max(
        1,
        Math.floor(
          width * renderScale
        )
      );

    canvas.height =
      Math.max(
        1,
        Math.floor(
          height * renderScale
        )
      );

    canvas.style.width =
      `${width}px`;

    canvas.style.height =
      `${height}px`;

    ctx.setTransform(
      renderScale,
      0,
      0,
      renderScale,
      0,
      0
    );

    backgroundGradient =
      ctx.createRadialGradient(
        width * 0.5,
        height * 0.05,
        0,
        width * 0.5,
        height * 0.05,
        height * 0.95
      );

    backgroundGradient
      .addColorStop(
        0,
        "rgba(64,188,230,0.048)"
      );

    backgroundGradient
      .addColorStop(
        0.42,
        "rgba(8,75,103,0.022)"
      );

    backgroundGradient
      .addColorStop(
        1,
        "rgba(0,0,0,0)"
      );

    buildWorld();

  }


  function scheduleResize() {

    clearTimeout(
      resizeTimer
    );

    resizeTimer =
      setTimeout(
        resize,
        120
      );

  }


  function buildWorld() {

    nodes.length = 0;
    particles.length = 0;

    const nodeCount =
      Math.min(
        MAX_NODES,
        Math.max(
          26,
          Math.floor(
            width / 32
          )
        )
      );

    const particleCount =
      Math.min(
        MAX_PARTICLES,
        Math.max(
          24,
          Math.floor(
            width / 34
          )
        )
      );

    for (
      let i = 0;
      i < nodeCount;
      i += 1
    ) {
      nodes.push(
        new Node()
      );
    }

    for (
      let i = 0;
      i < particleCount;
      i += 1
    ) {
      particles.push(
        new Particle()
      );
    }

    connectionCounts =
      new Uint8Array(
        nodeCount
      );

  }


  function drawBackground(
    config
  ) {

    if (
      !backgroundGradient
    ) {
      return;
    }

    ctx.globalAlpha =
      config.cyan;

    ctx.fillStyle =
      backgroundGradient;

    ctx.fillRect(
      0,
      0,
      width,
      height
    );

    ctx.globalAlpha = 1;

  }


  function drawCurrents(
    time,
    config
  ) {

    ctx.beginPath();

    const lineCount = 4;

    for (
      let i = 0;
      i < lineCount;
      i += 1
    ) {

      const baseY =
        height *
        (
          0.20 +
          i * 0.17
        );

      for (
        let x = -40;
        x <= width + 40;
        x += 30
      ) {

        const y =
          baseY +
          Math.sin(
            x * 0.005 +
            time * 0.00013 +
            i * 1.7
          ) *
          (
            5 +
            i * 1.4
          );

        if (x === -40) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

      }

    }

    ctx.strokeStyle =
      `rgba(
        86,
        205,
        240,
        ${
          0.011 +
          config.network *
          0.014
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
      const particle
      of particles
    ) {

      particle.update();

      ctx.moveTo(
        particle.x +
        particle.size,
        particle.y
      );

      ctx.arc(
        particle.x,
        particle.y,
        particle.size,
        0,
        Math.PI * 2
      );

    }

    ctx.fillStyle =
      `rgba(
        185,
        234,
        248,
        ${
          0.075 *
          config.particle
        }
      )`;

    ctx.fill();

  }


  function drawNetwork(
    time,
    config
  ) {

    const connectionDistance =
      Math.min(
        128,
        width * 0.095
      );

    const connectionDistanceSq =
      connectionDistance *
      connectionDistance;

    connectionCounts.fill(0);
    activeEdges.length = 0;

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
     * Draw the whole quiet network as one
     * path rather than one stroke call per
     * connection.
     */

    ctx.beginPath();

    for (
      let i = 0;
      i < nodes.length;
      i += 1
    ) {

      if (
        connectionCounts[i] >=
        MAX_CONNECTIONS
      ) {
        continue;
      }

      const a =
        nodes[i];

      for (
        let j = i + 1;
        j < nodes.length;
        j += 1
      ) {

        if (
          connectionCounts[i] >=
          MAX_CONNECTIONS
        ) {
          break;
        }

        if (
          connectionCounts[j] >=
          MAX_CONNECTIONS
        ) {
          continue;
        }

        const b =
          nodes[j];

        const dx =
          a.x - b.x;

        const dy =
          a.y - b.y;

        const distSq =
          dx * dx +
          dy * dy;

        if (
          distSq >
          connectionDistanceSq
        ) {
          continue;
        }

        connectionCounts[i] += 1;
        connectionCounts[j] += 1;

        ctx.moveTo(
          a.x,
          a.y
        );

        ctx.lineTo(
          b.x,
          b.y
        );

        const activity =
          Math.max(
            a.energy,
            b.energy
          );

        if (
          activity > 0.56
        ) {

          activeEdges.push({
            a,
            b,
            activity,
            seed:
              (
                i * 17 +
                j * 11
              ) *
              0.013
          });

        }

      }

    }

    ctx.strokeStyle =
      `rgba(
        74,
        207,
        245,
        ${
          0.10 *
          config.network
        }
      )`;

    ctx.lineWidth = 0.7;
    ctx.stroke();


    /*
     * Only the small number of active
     * connections receives the brighter
     * second pass.
     */

    if (
      activeEdges.length
    ) {

      ctx.beginPath();

      for (
        const edge
        of activeEdges
      ) {

        ctx.moveTo(
          edge.a.x,
          edge.a.y
        );

        ctx.lineTo(
          edge.b.x,
          edge.b.y
        );

      }

      ctx.strokeStyle =
        `rgba(
          105,
          226,
          255,
          ${
            0.22 *
            config.network
          }
        )`;

      ctx.lineWidth = 1.05;
      ctx.stroke();


      for (
        const edge
        of activeEdges
      ) {

        const travel =
          (
            time * 0.00018 +
            edge.seed
          ) % 1;

        const x =
          edge.a.x +
          (
            edge.b.x -
            edge.a.x
          ) *
          travel;

        const y =
          edge.a.y +
          (
            edge.b.y -
            edge.a.y
          ) *
          travel;

        ctx.beginPath();

        ctx.arc(
          x,
          y,
          1.1 +
          edge.activity * 0.8,
          0,
          Math.PI * 2
        );

        ctx.fillStyle =
          `rgba(
            126,
            235,
            255,
            ${
              0.22 +
              edge.activity *
              0.33
            }
          )`;

        ctx.fill();

      }

    }


    /*
     * Quiet nodes are batched.
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
        Math.PI * 2
      );

    }

    ctx.fillStyle =
      `rgba(
        105,
        224,
        255,
        ${
          0.17 *
          config.network
        }
      )`;

    ctx.fill();


    /*
     * Bright energy is rare, so individual
     * glows remain cheap.
     */

    for (
      const node
      of nodes
    ) {

      if (
        node.energy < 0.48
      ) {
        continue;
      }

      ctx.beginPath();

      ctx.arc(
        node.x,
        node.y,
        1.5 +
        node.energy * 2,
        0,
        Math.PI * 2
      );

      ctx.fillStyle =
        `rgba(
          119,
          231,
          255,
          ${
            node.energy *
            config.network *
            0.62
          }
        )`;

      ctx.fill();

    }

  }


  function render(
    time
  ) {

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

    drawBackground(
      config
    );

    drawCurrents(
      time,
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


  function animationLoop(
    time
  ) {

    if (!running) {
      return;
    }

    const fps =
      scrolling
        ? SCROLL_FPS
        : TARGET_FPS;

    const frameInterval =
      1000 / fps;

    if (
      time -
      lastFrameTime >=
      frameInterval
    ) {

      lastFrameTime =
        time -
        (
          (
            time -
            lastFrameTime
          ) %
          frameInterval
        );

      render(time);

    }

    requestAnimationFrame(
      animationLoop
    );

  }


  function updateScrollNow() {

    scrollQueued = false;

    const documentElement =
      document.documentElement;

    const maxScroll =
      documentElement.scrollHeight -
      window.innerHeight;

    const progress =
      maxScroll > 0
        ? window.scrollY /
          maxScroll
        : 0;

    progressBar.style.transform =
      `scaleX(${progress})`;

    scrolling = true;

    clearTimeout(
      scrollTimer
    );

    scrollTimer =
      setTimeout(
        () => {
          scrolling = false;
        },
        140
      );

    if (
      reducedMotion
    ) {
      render(
        performance.now()
      );
    }

  }


  function queueScrollUpdate() {

    if (scrollQueued) {
      return;
    }

    scrollQueued = true;

    requestAnimationFrame(
      updateScrollNow
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
              best.target
                .dataset
                .scene ||
              "surface";

          }

        },
        {
          threshold: [
            0.2,
            0.4,
            0.6
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

    if (
      reducedMotion
    ) {

      elements.forEach(
        element =>
          element
            .classList
            .add("visible")
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
          threshold: 0.10,
          rootMargin:
            "0px 0px -30px 0px"
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

        pointerX = -10000;
        pointerY = -10000;

      }
    );

  }


  function activateCluster(
    index,
    total
  ) {

    const start =
      Math.floor(
        index *
        nodes.length /
        total
      );

    const end =
      Math.floor(
        (
          index + 1
        ) *
        nodes.length /
        total
      );

    for (
      let i = start;
      i < end;
      i += 1
    ) {

      if (nodes[i]) {

        nodes[i].energy =
          0.84 +
          Math.random() *
          0.16;

      }

    }

  }


  function bindRuntimeMap() {

    const runtimeNodes =
      document.querySelectorAll(
        ".runtime-node"
      );

    runtimeNodes.forEach(
      (
        node,
        index
      ) => {

        node.addEventListener(
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


  function handleVisibility() {

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
      lastFrameTime = 0;

      requestAnimationFrame(
        animationLoop
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
    queueScrollUpdate,
    {
      passive: true
    }
  );

  document.addEventListener(
    "visibilitychange",
    handleVisibility
  );


  resize();
  queueScrollUpdate();
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
      animationLoop
    );

  }

})();
