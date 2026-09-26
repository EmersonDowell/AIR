(() => {
  "use strict";

  const canvas = document.getElementById("neural-ocean");
  const ctx = canvas.getContext("2d", { alpha: true });

  const progressBar =
    document.getElementById("scroll-progress-bar");

  const prefersReduced =
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let width = 0;
  let height = 0;
  let dpr = 1;

  let scrollProgress = 0;
  let activeScene = "surface";

  let pointerX = -10000;
  let pointerY = -10000;

  const nodes = [];
  const particles = [];

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
      this.reset(true);
    }

    reset(initial = false) {
      this.x = Math.random() * width;
      this.y = Math.random() * height;

      this.vx =
        (Math.random() - 0.5) *
        (initial ? 0.13 : 0.18);

      this.vy =
        (Math.random() - 0.5) *
        (initial ? 0.10 : 0.15);

      this.radius =
        0.7 + Math.random() * 1.6;

      this.energy = Math.random() * 0.25;

      this.phase =
        Math.random() * Math.PI * 2;
    }

    update(time, config) {
      this.x += this.vx;
      this.y += this.vy;

      if (this.x < -50) this.x = width + 50;
      if (this.x > width + 50) this.x = -50;

      if (this.y < -50) this.y = height + 50;
      if (this.y > height + 50) this.y = -50;

      const dx = pointerX - this.x;
      const dy = pointerY - this.y;

      const dist =
        Math.sqrt(dx * dx + dy * dy);

      if (dist < 180) {
        const influence =
          (1 - dist / 180) * 0.7;

        this.energy =
          Math.max(this.energy, influence);
      }

      const wave =
        Math.sin(time * 0.0012 + this.phase);

      if (
        wave > 0.985 &&
        Math.random() < config.pulse * 0.025
      ) {
        this.energy = 1;
      }

      this.energy *= 0.976;
    }
  }


  class Particle {
    constructor() {
      this.reset(true);
    }

    reset(initial = false) {
      this.x = Math.random() * width;

      this.y = initial
        ? Math.random() * height
        : height + Math.random() * 100;

      this.size =
        0.4 + Math.random() * 1.6;

      this.speed =
        0.10 + Math.random() * 0.32;

      this.drift =
        (Math.random() - 0.5) * 0.08;

      this.alpha =
        0.05 + Math.random() * 0.16;
    }

    update() {
      this.y -= this.speed;
      this.x += this.drift;

      if (
        this.y < -30 ||
        this.x < -50 ||
        this.x > width + 50
      ) {
        this.reset();
      }
    }
  }


  function resize() {
    dpr =
      Math.min(
        window.devicePixelRatio || 1,
        2
      );

    width = window.innerWidth;
    height = window.innerHeight;

    canvas.width =
      Math.floor(width * dpr);

    canvas.height =
      Math.floor(height * dpr);

    canvas.style.width =
      `${width}px`;

    canvas.style.height =
      `${height}px`;

    ctx.setTransform(
      dpr,
      0,
      0,
      dpr,
      0,
      0
    );

    buildWorld();
  }


  function buildWorld() {
    nodes.length = 0;
    particles.length = 0;

    const nodeCount =
      Math.min(
        95,
        Math.max(
          40,
          Math.floor(width / 17)
        )
      );

    const particleCount =
      Math.min(
        105,
        Math.max(
          40,
          Math.floor(width / 15)
        )
      );

    for (
      let i = 0;
      i < nodeCount;
      i += 1
    ) {
      nodes.push(new Node());
    }

    for (
      let i = 0;
      i < particleCount;
      i += 1
    ) {
      particles.push(new Particle());
    }
  }


  function drawBackgroundGlow(config) {
    const gradient =
      ctx.createRadialGradient(
        width * 0.5,
        height * 0.05,
        0,
        width * 0.5,
        height * 0.05,
        height * 0.95
      );

    gradient.addColorStop(
      0,
      `rgba(64, 188, 230, ${
        0.055 * config.cyan
      })`
    );

    gradient.addColorStop(
      0.42,
      "rgba(8, 75, 103, 0.025)"
    );

    gradient.addColorStop(
      1,
      "rgba(0, 0, 0, 0)"
    );

    ctx.fillStyle = gradient;

    ctx.fillRect(
      0,
      0,
      width,
      height
    );
  }


  function drawParticles(config) {
    for (const p of particles) {
      p.update();

      ctx.beginPath();

      ctx.arc(
        p.x,
        p.y,
        p.size,
        0,
        Math.PI * 2
      );

      ctx.fillStyle =
        `rgba(
          170,
          230,
          247,
          ${p.alpha * config.particle}
        )`;

      ctx.fill();
    }
  }


  function drawNetwork(time, config) {
    const connectionDistance =
      Math.min(150, width * 0.12);

    for (const n of nodes) {
      n.update(time, config);
    }

    for (
      let i = 0;
      i < nodes.length;
      i += 1
    ) {
      const a = nodes[i];

      for (
        let j = i + 1;
        j < nodes.length;
        j += 1
      ) {
        const b = nodes[j];

        const dx = a.x - b.x;
        const dy = a.y - b.y;

        const distSq =
          dx * dx + dy * dy;

        const maxSq =
          connectionDistance *
          connectionDistance;

        if (distSq > maxSq) {
          continue;
        }

        const dist =
          Math.sqrt(distSq);

        const proximity =
          1 - dist / connectionDistance;

        const activity =
          Math.max(a.energy, b.energy);

        const alpha =
          proximity *
          (
            0.055 +
            activity * 0.32
          ) *
          config.network;

        ctx.beginPath();

        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);

        ctx.strokeStyle =
          `rgba(
            74,
            207,
            245,
            ${alpha}
          )`;

        ctx.lineWidth =
          0.45 +
          activity * 0.95;

        ctx.stroke();

        if (activity > 0.63) {
          const travel =
            (
              time * 0.00022 +
              (i + j) * 0.07
            ) % 1;

          const sx =
            a.x + (b.x - a.x) * travel;

          const sy =
            a.y + (b.y - a.y) * travel;

          ctx.beginPath();

          ctx.arc(
            sx,
            sy,
            1.2 + activity * 1.2,
            0,
            Math.PI * 2
          );

          ctx.fillStyle =
            `rgba(
              120,
              232,
              255,
              ${
                0.16 +
                activity * 0.48
              }
            )`;

          ctx.fill();
        }
      }
    }

    for (const n of nodes) {
      const glow =
        0.10 +
        n.energy * 0.70;

      ctx.beginPath();

      ctx.arc(
        n.x,
        n.y,
        n.radius +
          n.energy * 1.7,
        0,
        Math.PI * 2
      );

      ctx.fillStyle =
        `rgba(
          105,
          224,
          255,
          ${
            glow *
            config.network *
            0.65
          }
        )`;

      ctx.fill();

      if (n.energy > 0.58) {
        ctx.beginPath();

        ctx.arc(
          n.x,
          n.y,
          7 + n.energy * 10,
          0,
          Math.PI * 2
        );

        ctx.fillStyle =
          `rgba(
            75,
            215,
            255,
            ${
              n.energy *
              config.network *
              0.025
            }
          )`;

        ctx.fill();
      }
    }
  }


  function drawCurrentLines(time, config) {
    const lineCount = 5;

    for (
      let i = 0;
      i < lineCount;
      i += 1
    ) {
      const y =
        height *
        (
          0.17 +
          i * 0.14
        );

      ctx.beginPath();

      for (
        let x = -50;
        x <= width + 50;
        x += 18
      ) {
        const wave =
          Math.sin(
            x * 0.006 +
            time * 0.00018 +
            i * 1.4
          );

        const drift =
          Math.sin(
            time * 0.00007 +
            i
          ) * 20;

        const py =
          y +
          wave * (6 + i * 1.5) +
          drift;

        if (x === -50) {
          ctx.moveTo(x, py);
        } else {
          ctx.lineTo(x, py);
        }
      }

      ctx.strokeStyle =
        `rgba(
          86,
          205,
          240,
          ${
            0.014 +
            config.network * 0.018
          }
        )`;

      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }


  function animate(time) {
    const config =
      sceneConfig[activeScene] ||
      sceneConfig.surface;

    ctx.clearRect(
      0,
      0,
      width,
      height
    );

    drawBackgroundGlow(config);
    drawCurrentLines(time, config);
    drawParticles(config);
    drawNetwork(time, config);

    if (!prefersReduced) {
      requestAnimationFrame(animate);
    }
  }


  function updateScroll() {
    const doc =
      document.documentElement;

    const max =
      doc.scrollHeight -
      window.innerHeight;

    scrollProgress =
      max > 0
        ? window.scrollY / max
        : 0;

    progressBar.style.width =
      `${scrollProgress * 100}%`;
  }


  function observeScenes() {
    const scenes =
      document.querySelectorAll(".scene");

    const observer =
      new IntersectionObserver(
        entries => {
          let best = null;

          for (const entry of entries) {
            if (!entry.isIntersecting) {
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
            0.2,
            0.35,
            0.5,
            0.65
          ]
        }
      );

    scenes.forEach(
      scene => observer.observe(scene)
    );
  }


  function observeReveals() {
    const reveals =
      document.querySelectorAll(".reveal");

    if (prefersReduced) {
      reveals.forEach(
        el => el.classList.add("visible")
      );

      return;
    }

    const observer =
      new IntersectionObserver(
        entries => {
          for (const entry of entries) {
            if (entry.isIntersecting) {
              entry.target.classList.add(
                "visible"
              );

              observer.unobserve(
                entry.target
              );
            }
          }
        },
        {
          threshold: 0.12,
          rootMargin:
            "0px 0px -40px 0px"
        }
      );

    reveals.forEach(
      el => observer.observe(el)
    );
  }


  function bindPointer() {
    window.addEventListener(
      "pointermove",
      event => {
        pointerX = event.clientX;
        pointerY = event.clientY;
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


  function bindRuntimeMap() {
    const runtimeNodes =
      document.querySelectorAll(
        ".runtime-node"
      );

    runtimeNodes.forEach(
      (node, index) => {
        node.addEventListener(
          "pointerenter",
          () => {
            const start =
              Math.floor(
                index *
                nodes.length /
                runtimeNodes.length
              );

            const end =
              Math.floor(
                (index + 1) *
                nodes.length /
                runtimeNodes.length
              );

            for (
              let i = start;
              i < end;
              i += 1
            ) {
              if (nodes[i]) {
                nodes[i].energy =
                  0.85 +
                  Math.random() * 0.15;
              }
            }
          }
        );
      }
    );
  }


  window.addEventListener(
    "resize",
    resize,
    {
      passive: true
    }
  );

  window.addEventListener(
    "scroll",
    updateScroll,
    {
      passive: true
    }
  );

  resize();
  updateScroll();
  observeScenes();
  observeReveals();
  bindPointer();
  bindRuntimeMap();

  if (prefersReduced) {
    animate(0);
  } else {
    requestAnimationFrame(animate);
  }
})();
