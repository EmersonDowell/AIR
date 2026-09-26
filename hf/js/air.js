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
   * Keep the optimized rendering envelope
   * from the previous performance pass.
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
      network: 0.23,
      pulse: 0.18,
      particles: 0.28,
      currents: 0.25,
      wobble: 7
    },

    shallows: {
      network: 0.32,
      pulse: 0.25,
      particles: 0.34,
      currents: 0.38,
      wobble: 6
    },

    network: {
      network: 0.68,
      pulse: 0.72,
      particles: 0.18,
      currents: 0.12,
      wobble: 2.5
    },

    deep: {
      network: 0.46,
      pulse: 0.42,
      particles: 0.21,
      currents: 0.16,
      wobble: 4
    },

    signal: {
      network: 0.74,
      pulse: 0.90,
      particles: 0.16,
      currents: 0.12,
      wobble: 2
    },

    horizon: {
      network: 0.29,
      pulse: 0.24,
      particles: 0.28,
      currents: 0.17,
      wobble: 3
    }

  };


  function clamp(
    value,
    min,
    max
  ) {

    return Math.max(
      min,
      Math.min(
        max,
        value
      )
    );

  }


  class Node {

    constructor(index) {

      this.index = index;

      /*
       * Stable random identity.
       * These values never change after creation.
       */

      this.seedX =
        Math.random();

      this.seedY =
        Math.random();

      this.phase =
        Math.random() *
        Math.PI *
        2;

      this.phaseY =
        Math.random() *
        Math.PI *
        2;

      this.radius =
        0.85 +
        Math.random() *
        1.15;

      this.energy =
        Math.random() *
        0.18;

      this.x =
        this.seedX *
        width;

      this.y =
        this.seedY *
        height;

      this.tx = this.x;
      this.ty = this.y;

    }


    setTarget(
      normalizedX,
      normalizedY,
      immediate = false
    ) {

      this.tx =
        clamp(
          normalizedX,
          0.035,
          0.965
        ) *
        width;

      this.ty =
        clamp(
          normalizedY,
          0.055,
          0.945
        ) *
        height;

      if (immediate) {

        this.x =
          this.tx;

        this.y =
          this.ty;

      }

    }


    update(
      time,
      config
    ) {

      const wobble =
        config.wobble;

      const goalX =
        this.tx +
        Math.sin(
          time * 0.00018 +
          this.phase
        ) *
        wobble;

      const goalY =
        this.ty +
        Math.sin(
          time * 0.00014 +
          this.phaseY
        ) *
        wobble *
        0.72;

      /*
       * Smooth morphing between environmental
       * and computational arrangements.
       */

      this.x +=
        (
          goalX -
          this.x
        ) *
        0.075;

      this.y +=
        (
          goalY -
          this.y
        ) *
        0.075;


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

        const radius = 165;
        const radiusSq =
          radius *
          radius;

        if (
          distanceSq <
          radiusSq
        ) {

          const influence =
            (
              1 -
              distanceSq /
              radiusSq
            ) *
            0.60;

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
          time * 0.00082 +
          this.phase
        ) > 0.995 &&
        Math.random() <
          config.pulse *
          0.014
      ) {

        this.energy = 1;

      }

      this.energy *= 0.969;

    }

  }


  class Particle {

    constructor() {
      this.reset(true);
    }


    reset(
      initial = false
    ) {

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


  function targetForScene(
    scene,
    index,
    node
  ) {

    const count =
      nodes.length ||
      NODE_COUNT;


    /*
     * HERO
     *
     * Organic neural field suspended in water.
     */

    if (
      scene === "surface"
    ) {

      return {
        x:
          node.seedX,

        y:
          node.seedY
      };

    }


    /*
     * SHALLOWS
     *
     * Neural activity begins to resemble
     * horizontal Lake Superior currents.
     */

    if (
      scene === "shallows"
    ) {

      const lanes = 4;

      const lane =
        index %
        lanes;

      const slot =
        Math.floor(
          index /
          lanes
        );

      const slots =
        Math.ceil(
          count /
          lanes
        );

      const x =
        (
          slot +
          0.45
        ) /
        Math.max(
          slots,
          1
        );

      const y =
        0.21 +
        lane *
        0.18 +
        Math.sin(
          index *
          1.13
        ) *
        0.024;

      return {
        x:
          x +
          Math.sin(
            node.phase
          ) *
          0.015,

        y
      };

    }


    /*
     * ARCHITECTURE
     *
     * Five distinct vertical stages form:
     *
     * Request
     * Runtime
     * Inference
     * Decision
     * State
     */

    if (
      scene === "network"
    ) {

      const columns = 5;

      const column =
        index %
        columns;

      const row =
        Math.floor(
          index /
          columns
        );

      const rowsInColumn =
        Math.ceil(
          (
            count -
            column
          ) /
          columns
        );

      const x =
        0.11 +
        column *
        0.195;

      const y =
        rowsInColumn <= 1
          ? 0.50
          : 0.18 +
            (
              row /
              (
                rowsInColumn -
                1
              )
            ) *
            0.64;

      return {
        x,
        y:
          y +
          (
            node.seedY -
            0.5
          ) *
          0.018
      };

    }


    /*
     * RESEARCH
     *
     * The clean graph separates into several
     * experimental clusters.
     */

    if (
      scene === "deep"
    ) {

      const centers = [
        [0.27, 0.31],
        [0.69, 0.28],
        [0.35, 0.70],
        [0.74, 0.68]
      ];

      const cluster =
        index %
        centers.length;

      const ring =
        Math.floor(
          index /
          centers.length
        );

      const angle =
        node.phase +
        ring *
        0.72;

      const radius =
        0.035 +
        (
          ring %
          4
        ) *
        0.013;

      return {
        x:
          centers[cluster][0] +
          Math.cos(
            angle
          ) *
          radius,

        y:
          centers[cluster][1] +
          Math.sin(
            angle
          ) *
          radius *
          1.35
      };

    }


    /*
     * RELEASE
     *
     * Reassemble into a dense directed
     * signal-processing graph.
     */

    if (
      scene === "signal"
    ) {

      const columns = 5;

      const column =
        index %
        columns;

      const row =
        Math.floor(
          index /
          columns
        );

      const rows =
        Math.ceil(
          (
            count -
            column
          ) /
          columns
        );

      const centeredRow =
        row -
        (
          rows -
          1
        ) /
        2;

      return {
        x:
          0.09 +
          column *
          0.205,

        y:
          0.50 +
          centeredRow *
          0.075 *
          (
            1 -
            column *
            0.045
          ) +
          Math.sin(
            node.phase
          ) *
          0.012
      };

    }


    /*
     * FINALE
     *
     * Half of the graph becomes a luminous
     * horizon. The rest disperses upward into
     * a constellation field.
     */

    if (
      scene === "horizon"
    ) {

      const horizonCount =
        Math.max(
          8,
          Math.floor(
            count *
            0.46
          )
        );

      if (
        index <
        horizonCount
      ) {

        const t =
          (
            index +
            1
          ) /
          (
            horizonCount +
            1
          );

        return {
          x: t,

          y:
            0.61 -
            Math.sin(
              t *
              Math.PI
            ) *
            0.065
        };

      }

      return {
        x:
          node.seedX,

        y:
          0.10 +
          node.seedY *
          0.34
      };

    }


    return {
      x:
        node.seedX,

      y:
        node.seedY
    };

  }


  function applySceneTargets(
    scene,
    immediate = false
  ) {

    for (
      let i = 0;
      i < nodes.length;
      i += 1
    ) {

      const target =
        targetForScene(
          scene,
          i,
          nodes[i]
        );

      nodes[i].setTarget(
        target.x,
        target.y,
        immediate
      );

    }

  }


  function addEdge(
    a,
    b
  ) {

    if (
      a === b ||
      a < 0 ||
      b < 0 ||
      a >= nodes.length ||
      b >= nodes.length
    ) {
      return;
    }

    const exists =
      edges.some(
        edge =>
          (
            edge.a === a &&
            edge.b === b
          ) ||
          (
            edge.a === b &&
            edge.b === a
          )
      );

    if (exists) {
      return;
    }

    edges.push({
      a,
      b,
      seed:
        Math.random()
    });

  }


  function buildOrganicEdges() {

    edges.length = 0;

    const targets =
      nodes.map(
        node => ({
          x:
            node.tx /
            width,

          y:
            node.ty /
            height
        })
      );

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

        if (
          i === j
        ) {
          continue;
        }

        const dx =
          targets[i].x -
          targets[j].x;

        const dy =
          targets[i].y -
          targets[j].y;

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

        addEdge(
          i,
          nearest[k].j
        );

      }

    }

  }


  function buildFlowEdges(
    dense = false
  ) {

    edges.length = 0;

    const columns = 5;

    /*
     * Primary forward paths.
     */

    for (
      let i = 0;
      i < nodes.length;
      i += 1
    ) {

      const column =
        i %
        columns;

      if (
        column >=
        columns - 1
      ) {
        continue;
      }

      const next =
        i + 1;

      if (
        next <
        nodes.length
      ) {

        addEdge(
          i,
          next
        );

      }

      /*
       * Occasional cross-row routes make the
       * runtime look like a real computational
       * graph rather than five straight wires.
       */

      if (
        (
          i %
          3 === 0 ||
          dense
        ) &&
        i + 6 <
        nodes.length
      ) {

        addEdge(
          i,
          i + 6
        );

      }

    }


    /*
     * Sparse within-stage relationships.
     */

    for (
      let i = 0;
      i + 5 < nodes.length;
      i += 2
    ) {

      addEdge(
        i,
        i + 5
      );

    }

  }


  function rebuildEdges(
    scene
  ) {

    if (
      scene === "network"
    ) {

      buildFlowEdges(
        false
      );

      return;

    }

    if (
      scene === "signal"
    ) {

      buildFlowEdges(
        true
      );

      return;

    }

    buildOrganicEdges();

  }


  function setScene(
    scene,
    immediate = false
  ) {

    if (
      scene === activeScene &&
      !immediate
    ) {
      return;
    }

    activeScene = scene;

    applySceneTargets(
      activeScene,
      immediate
    );

    rebuildEdges(
      activeScene
    );


    /*
     * Scene transitions wake a small number
     * of nodes so the topology seems to
     * "come online."
     */

    if (!immediate) {

      for (
        let i = 0;
        i < nodes.length;
        i += 5
      ) {

        nodes[i].energy =
          Math.max(
            nodes[i].energy,
            0.58
          );

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


  function buildWorld() {

    nodes.length = 0;
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

    setScene(
      activeScene,
      true
    );

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

    buildWorld();

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
     * Water currents become calmer as the
     * page becomes more computational.
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

        if (
          x === -40
        ) {

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
          0.045 *
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
     * Main topology.
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


    if (
      activeScene ===
      "horizon"
    ) {

      ctx.strokeStyle =
        `rgba(
          91,
          201,
          232,
          ${
            0.16 *
            config.network
          }
        )`;

    } else {

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

    }

    ctx.lineWidth =
      activeScene ===
      "network" ||
      activeScene ===
      "signal"
        ? 0.95
        : 0.75;

    ctx.stroke();


    /*
     * Cyan neural points.
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
     * At the horizon the computational field
     * begins turning into a constellation.
     * A few points pick up the warm release-art
     * horizon color.
     */

    if (
      activeScene ===
      "horizon"
    ) {

      ctx.beginPath();

      for (
        let i = 2;
        i < nodes.length;
        i += 6
      ) {

        const node =
          nodes[i];

        ctx.moveTo(
          node.x + 1.5,
          node.y
        );

        ctx.arc(
          node.x,
          node.y,
          1.5,
          0,
          Math.PI *
          2
        );

      }

      ctx.fillStyle =
        "rgba(231,168,95,0.62)";

      ctx.fill();

    }


    /*
     * Traveling activation signals.
     */

    let pulses = 0;

    const pulseLimit =
      activeScene === "signal"
        ? 10
        : activeScene === "network"
          ? 8
          : 6;

    for (
      const edge
      of edges
    ) {

      if (
        pulses >=
        pulseLimit
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
        energy <
        0.46
      ) {
        continue;
      }

      const directionSpeed =
        activeScene ===
        "signal"
          ? 0.00022
          : 0.00015;

      const t =
        (
          time *
          directionSpeed +
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


      /*
       * Tiny trailing segment gives the signal
       * a directional feeling without shaders.
       */

      const tail =
        Math.max(
          0,
          t - 0.055
        );

      const tx =
        a.x +
        (
          b.x -
          a.x
        ) *
        tail;

      const ty =
        a.y +
        (
          b.y -
          a.y
        ) *
        tail;

      ctx.beginPath();

      ctx.moveTo(
        tx,
        ty
      );

      ctx.lineTo(
        x,
        y
      );

      ctx.strokeStyle =
        `rgba(
          115,
          232,
          255,
          ${
            0.16 +
            energy *
            0.25
          }
        )`;

      ctx.lineWidth =
        1.1;

      ctx.stroke();


      ctx.beginPath();

      ctx.arc(
        x,
        y,
        1.35 +
        energy *
        0.8,
        0,
        Math.PI *
        2
      );

      ctx.fillStyle =
        `rgba(
          139,
          241,
          255,
          ${
            0.34 +
            energy *
            0.38
          }
        )`;

      ctx.fill();

      pulses += 1;

    }

  }


  function render(
    time
  ) {

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


  function loop(
    time
  ) {

    if (!running) {
      return;
    }

    /*
     * Preserve the important performance fix:
     * freeze the atmospheric frame while the
     * browser is physically scrolling.
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

      render(
        time
      );

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

    if (
      progressBar
    ) {

      progressBar
        .style
        .transform =
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

          render(
            performance.now()
          );

        },
        110
      );

  }


  function queueScroll() {

    if (
      scrollQueued
    ) {
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

              best =
                entry;

            }

          }

          if (
            best
          ) {

            const nextScene =
              best.target
                .dataset
                .scene ||
              "surface";

            if (
              nextScene !==
              activeScene
            ) {

              setScene(
                nextScene
              );

            }

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

    if (
      reducedMotion
    ) {

      elements.forEach(
        element =>
          element
            .classList
            .add(
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
              .add(
                "visible"
              );

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

        pointerX =
          -99999;

        pointerY =
          -99999;

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
        0.88 +
        Math.random() *
        0.12;

    }

  }


  function bindRuntimeMap() {

    const runtimeNodes =
      document.querySelectorAll(
        ".runtime-node"
      );

    runtimeNodes.forEach(
      (
        element,
        index
      ) => {

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
