// GPU vertex i always corresponds to row i in the source PLY.
"use strict";
const Scene3D = (() => {
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const scale = (a, s) => a.map((v) => v * s);
  const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a) => scale(a, 1 / Math.hypot(...a));
  const rotate = (v, axis, angle) =>
    add(
      add(scale(v, Math.cos(angle)), scale(cross(axis, v), Math.sin(angle))),
      scale(axis, dot(axis, v) * (1 - Math.cos(angle))),
    );
  const field = 0.45; // tangent of half the vertical field of view
  const label = (id) => document.getElementById(id);

  function program(gl) {
    const sources = [
      [
        gl.VERTEX_SHADER,
        `#version 300 es
        layout(location=0) in vec3 position;
        uniform mat3 basis;
        uniform vec3 eye;
        uniform vec4 lens;
        uniform float size;
        void main(){
          vec3 p = basis*(position-eye);
          gl_Position = vec4(p.xy*lens.xy,p.z*lens.z+lens.w,p.z);
          gl_PointSize = size;
        }`,
      ],
      [
        gl.FRAGMENT_SHADER,
        `#version 300 es
        precision mediump float;
        uniform vec3 color;
        out vec4 pixel;
        void main(){pixel=vec4(color,1.0);}`,
      ],
    ];
    const prog = gl.createProgram();
    for (const [type, source] of sources) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
      gl.attachShader(prog, shader);
      gl.deleteShader(shader);
    }
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(prog));
    return prog;
  }

  return class Scene3D {
    constructor(canvas, info) {
      this.canvas = canvas;
      this.info = info;
      this.gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
      if (!this.gl)
        throw Error("The 3D view requires WebGL 2. Enable graphics acceleration in this browser.");
      const gl = this.gl;
      const prog = program(gl);
      gl.useProgram(prog);
      this.u = Object.fromEntries(
        ["basis", "eye", "lens", "size", "color"].map((k) => [k, gl.getUniformLocation(prog, k)]),
      );
      this.cloud = gl.createBuffer();
      this.rows = gl.createBuffer();
      this.other = gl.createBuffer();
      this.frustum = gl.createBuffer();
      this.loaded = 0;
      this.selectedCount = 0;
      this.otherCount = 0;
      this.lines = 0;
      this.up = [0, 0, 1];
      this.back = unit([0.7, -1, 0.65]);
      this.target = [0, 0, 0];
      this.span = Math.max(1e-3, Math.hypot(...sub(info.bounds[1], info.bounds[0])));
      this.fit();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cloud);
      gl.bufferData(gl.ARRAY_BUFFER, 12 * info.ply_rows, gl.STATIC_DRAW);
      if (gl.getError() !== gl.NO_ERROR)
        throw Error("The browser could not allocate the full LiDAR point buffer.");
      gl.enableVertexAttribArray(0);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      this.controls();
      canvas.addEventListener("webglcontextlost", () => {
        label("cloudState").textContent = "Graphics context lost. Reload the viewer to restore the scene.";
      });
      new ResizeObserver(() => this.queueDraw()).observe(canvas);
      this.loadCloud().catch((e) => (label("cloudState").textContent = e.message));
    }

    async loadCloud() {
      const total = this.info.ply_rows,
        step = this.info.chunk_rows,
        gl = this.gl;
      if (!total) {
        label("cloudState").textContent = "No PLY vertices available.";
        return;
      }
      for (let start = 0; start < total; start += step) {
        const response = await fetch("/cloud/" + start / step);
        if (!response.ok) throw Error("Could not load LiDAR rows starting at " + start.toLocaleString());
        const bytes = await response.arrayBuffer(),
          count = Math.min(step, total - start);
        if (bytes.byteLength !== count * 12) throw Error("Incomplete LiDAR chunk at row " + start);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.cloud);
        gl.bufferSubData(gl.ARRAY_BUFFER, start * 12, new Float32Array(bytes));
        if (gl.getError() !== gl.NO_ERROR)
          throw Error("The browser could not upload LiDAR rows starting at " + start);
        this.loaded = start + count;
        label("cloudState").textContent =
          "LiDAR " + this.loaded.toLocaleString() + " / " + total.toLocaleString();
        this.queueDraw();
      }
    }

    clearSelection() {
      this.selectionRequest?.abort();
      this.rowsReady = false;
      this.camera = null;
      this.selectedCount = this.otherCount = this.lines = 0;
      label("showOther").disabled = true;
      label("otherCount").textContent = "0";
      label("indexState").textContent = "Reading camera indices…";
      this.queueDraw();
    }

    async select(frame) {
      const request = (this.selectionRequest = new AbortController());
      const gl = this.gl;
      this.camera = frame.geometry;
      label("cameraLegend").textContent = frame.can_project
        ? "Cyan · selected camera"
        : "Cyan · nominal frustum (distortion omitted)";
      if (!this.oriented) {
        this.up = unit(this.camera.up);
        const right = unit(cross(this.camera.forward, this.up));
        this.back = unit(add(add(scale(this.camera.forward, -1), scale(right, 0.55)), scale(this.up, 0.65)));
        this.oriented = true;
      }
      this.updateFrustum();
      this.queueDraw();
      try {
        const res = await fetch("/indexed/" + frame.id, { signal: request.signal });
        if (!res.ok) throw Error("Could not load per-camera indices");
        const bytes = await res.arrayBuffer();
        if (request !== this.selectionRequest || request.signal.aborted) return;
        const [rows, others] = new Uint32Array(bytes, 0, 2);
        if (
          rows !== frame.indexed_rows ||
          others !== frame.unmatched_xyz ||
          bytes.byteLength !== 8 + 4 * rows + 12 * others
        )
          throw Error("Incomplete per-camera point data");
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.rows);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(bytes, 8, rows), gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.other);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bytes, 8 + rows * 4, others * 3), gl.DYNAMIC_DRAW);
        if (gl.getError() !== gl.NO_ERROR)
          throw Error("The browser could not upload this camera's point selection.");
        this.selectedCount = rows;
        this.otherCount = others;
        this.rowsReady = true;
        label("indexState").textContent = rows.toLocaleString() + " red LiDAR rows · " + frame.indexed_source;
        label("showOther").disabled = !others;
        label("otherCount").textContent = others.toLocaleString();
        this.queueDraw();
      } catch (e) {
        if (!request.signal.aborted) label("indexState").textContent = e.message;
      }
    }

    fit() {
      this.target = [0, 0, 0]; // coordinates sent by the server are centered on the full scene
      const aspect = Math.max(0.1, this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight));
      this.distance = (this.span * 0.56) / Math.sin(Math.atan(field * Math.min(1, aspect)));
      this.queueDraw();
    }
    focus() {
      if (!this.camera) return;
      const f = this.camera,
        len = this.frustumSize;
      this.up = unit(f.up);
      this.back = unit(
        add(add(scale(f.forward, -1), scale(unit(cross(f.forward, f.up)), 0.8)), scale(f.up, 0.5)),
      );
      this.target = add(f.center, scale(f.forward, len * 0.8));
      this.distance = 7 * len;
      this.queueDraw();
    }
    updateFrustum() {
      const { center, rays } = this.camera;
      this.frustumSize = this.span * 0.035;
      const corners = rays.map((r) => add(center, scale(r, this.frustumSize)));
      const lines = [];
      for (let i = 0; i < 4; i++) lines.push(center, corners[i], corners[i], corners[(i + 1) % 4]);
      const top = scale(add(corners[0], corners[1]), 0.5);
      lines.push(
        corners[0],
        add(top, scale(this.camera.up, this.frustumSize * 0.23)),
        add(top, scale(this.camera.up, this.frustumSize * 0.23)),
        corners[1],
      );
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.frustum);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(lines.flat()), this.gl.DYNAMIC_DRAW);
      this.lines = lines.length;
    }
    queueDraw() {
      if (this.pending) return;
      this.pending = true;
      requestAnimationFrame(() => {
        this.pending = false;
        this.draw();
      });
    }
    draw() {
      const gl = this.gl,
        c = this.canvas,
        dpr = Math.min(devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(c.clientWidth * dpr)),
        h = Math.max(1, Math.round(c.clientHeight * dpr));
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      }
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.035, 0.045, 0.055, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const right = unit(cross(this.up, this.back)),
        up = cross(this.back, right),
        forward = scale(this.back, -1);
      const eye = add(this.target, scale(this.back, this.distance));
      const near = Math.max(this.span * 1e-6, this.distance * 0.0005);
      const far = Math.max(
        2 * near,
        Math.hypot(...eye) + this.span * 2,
        this.camera ? Math.hypot(...sub(eye, this.camera.center)) + this.span : 0,
      );
      gl.uniformMatrix3fv(
        this.u.basis,
        false,
        new Float32Array([0, 1, 2].flatMap((i) => [right[i], up[i], forward[i]])),
      );
      gl.uniform3fv(this.u.eye, eye);
      gl.uniform4f(
        this.u.lens,
        h / (w * field),
        1 / field,
        (far + near) / (far - near),
        (-2 * near * far) / (far - near),
      );
      const layer = (buffer, color, size) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        gl.uniform3fv(this.u.color, color);
        gl.uniform1f(this.u.size, size * dpr);
      };
      gl.enable(gl.DEPTH_TEST);
      layer(this.cloud, [0.55, 0.55, 0.55], 1);
      gl.drawArrays(gl.POINTS, 0, this.loaded);
      if (label("through").checked) gl.disable(gl.DEPTH_TEST);
      if (this.rowsReady && this.loaded === this.info.ply_rows && this.selectedCount) {
        layer(this.cloud, [1, 0.08, 0.1], 4);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.rows);
        gl.drawElements(gl.POINTS, this.selectedCount, gl.UNSIGNED_INT, 0);
      }
      if (this.otherCount && label("showOther").checked) {
        layer(this.other, [1, 0.67, 0.12], 3);
        gl.drawArrays(gl.POINTS, 0, this.otherCount);
      }
      if (this.lines) {
        gl.disable(gl.DEPTH_TEST);
        layer(this.frustum, [0.15, 0.9, 1], 1);
        gl.drawArrays(gl.LINES, 0, this.lines);
      }
      gl.enable(gl.DEPTH_TEST);
    }
    controls() {
      const c = this.canvas;
      const pointers = new Map();
      const zoom = (factor) => {
        this.distance = Math.max(this.span * 0.00005, Math.min(this.span * 100, this.distance * factor));
      };
      const pan = (dx, dy) => {
        const right = unit(cross(this.up, this.back)),
          up = cross(this.back, right);
        this.target = add(
          this.target,
          scale(add(scale(right, -dx), scale(up, dy)), (2 * field * this.distance) / c.clientHeight),
        );
      };
      c.oncontextmenu = (e) => e.preventDefault();
      c.onpointerdown = (e) => {
        c.setPointerCapture(e.pointerId);
        pointers.set(e.pointerId, [e.clientX, e.clientY]);
      };
      c.onpointerup = c.onpointercancel = c.onlostpointercapture = (e) => pointers.delete(e.pointerId);
      c.onpointermove = (e) => {
        if (!pointers.has(e.pointerId)) return;
        const old = pointers.get(e.pointerId),
          now = [e.clientX, e.clientY],
          dx = now[0] - old[0],
          dy = now[1] - old[1];
        pointers.set(e.pointerId, now);
        if (pointers.size === 2) {
          const other = Array.from(pointers).find(([id]) => id !== e.pointerId)[1];
          const previous = Math.hypot(...sub(old, other)),
            current = Math.hypot(...sub(now, other));
          if (previous > 0 && current > 0) zoom(previous / current);
          pan(dx / 2, dy / 2);
        } else if (e.shiftKey || e.buttons & 6) pan(dx, dy);
        else {
          this.back = unit(rotate(this.back, this.up, -dx * 0.008));
          const pitch = Math.asin(Math.max(-1, Math.min(1, dot(this.back, this.up))));
          const target = Math.min(1.54, Math.max(-1.54, pitch + dy * 0.008));
          this.back = unit(rotate(this.back, unit(cross(this.up, this.back)), pitch - target));
        }
        this.queueDraw();
      };
      c.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          zoom(Math.exp(Math.max(-1, Math.min(1, e.deltaY * 0.0015))));
          this.queueDraw();
        },
        { passive: false },
      );
      label("through").onchange = label("showOther").onchange = () => this.queueDraw();
      label("fitScene").onclick = () => this.fit();
      label("focusCamera").onclick = () => this.focus();
    }
  };
})();
