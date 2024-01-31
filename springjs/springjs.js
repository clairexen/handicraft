/*
 *  SpringJS - a JavaScript mass-spring-system simulator
 *
 *  Copyright (C) 2012  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation; either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

function SpringJS(canvas, x_min, x_max, y_min, y_max)
{
	this.canvas = canvas;
	this.x_min = x_min;
	this.x_max = x_max;
	this.y_min = y_min;
	this.y_max = y_max;

	this.w = canvas.width;
	this.h = canvas.height;
	this.ctx = canvas.getContext('2d');

	this.ctx_unmap = function() {
		this.ctx.setTransform(1, 0, 0, 1, 1, 1);
	}

	this.ctx_map = function() {
		this.ctx.setTransform(1, 0, 0, 1, 1, 1);
		this.ctx.scale(this.w/(this.x_max-this.x_min), -this.h/(this.y_max-this.y_min));
		this.ctx.translate(-this.x_min, -this.y_max);
	}

	// parameters: m (mass), d (damping)
	// optional parameters: fx fy (fixed x/y ordinate)
	// variables: x, y, vx, vy, ax, ay
	this.masses = [];

	// parameters: m1 m2 (mass indices), k (spring constant), s (normal length)
	this.springs = [];

	// time and time step
	this.time = 0;
	this.dtime = 0.1;

	// max. frames per second and steps per second
	this.fps = 24;
	this.sps = 250;

	// step and frame counter for synchronization
	this.step_counter = 0;
	this.frame_counter = 0;

	// actual fps calculation
	this.actual_frame_count = 0;
	this.actual_step_count = 0;
	this.actual_start_time = 0;

	// display setting
	this.trace_mode = 2;
	this.trace_alpha = 0.1;
	this.show_fps = true;
	this.show_energy = false;
	this.color_time = 0;
	this.grid_step = 0;

	this.addMass = function(x, y, m, d)
	{
		var idx = this.masses.length;
		this.masses[idx] = {
			m: m, d: d, x: x, y: y,
			vx: 0, vy: 0, ax: 0, ay: 0
		};
		return idx;
	};

	this.addSpring = function(m1, m2, k, s)
	{
		var idx = this.springs.length;
		this.springs[idx] = {
			m1: m1, m2: m2, k: k, s: s
		};
		return idx;
	};

	// overloaded by user
	this.setup = function() { };
	this.update = function() { };

	this.calc = function()
	{
		if (this.time == 0) {
			this.masses = [];
			this.springs = [];
			this.setup();
		}

		for (var i = 0; i < this.masses.length; i++) {
			var M = this.masses[i];
			M.ax = 0, M.ay = 0;
		}

		this.update();

		for (var i = 0; i < this.springs.length; i++) {
			var S = this.springs[i];
			var M1 = this.masses[S.m1], M2 = this.masses[S.m2];
			var M1x = M1.x + M1.vx*this.dtime/2, M1y = M1.y + M1.vy*this.dtime/2;
			var M2x = M2.x + M2.vx*this.dtime/2, M2y = M2.y + M2.vy*this.dtime/2;
			var distance = Math.sqrt(Math.pow(M2x-M1x, 2) + Math.pow(M2y-M1y, 2));
			var fx = (M2x-M1x)/distance, fy = (M2y-M1y)/distance, f = (distance-S.s)*S.k;
			M1.ax += fx*f/M1.m, M1.ay += fy*f/M1.m;
			M2.ax -= fx*f/M2.m, M2.ay -= fy*f/M2.m;
		}

		for (var i = 0; i < this.masses.length; i++) {
			var M = this.masses[i];
			var damp = Math.exp(-M.d*this.dtime);
			M.x += M.vx*this.dtime + M.ax*this.dtime*this.dtime/2;
			M.y += M.vy*this.dtime + M.ay*this.dtime*this.dtime/2;
			M.vx = (M.vx + M.ax*this.dtime) * damp;
			M.vy = (M.vy + M.ay*this.dtime) * damp;
			if ("fx" in M)
				M.x = M.fx, M.vx = 0;
			if ("fy" in M)
				M.y = M.fy, M.vy = 0;
		}

		this.time += this.dtime;
	};

	this.clear_screen = function()
	{
		this.ctx_unmap();
		this.ctx.fillStyle = "rgb(0,0,0)";
		this.ctx.fillRect(0, 0, this.w, this.h);
		this.old_image = null;
	};

	this.draw_fps = function()
	{
		if (!this.show_fps || this.actual_frame_count < 5)
			return;
		
		var date = new Date();
		var ms = date.valueOf() - this.actual_start_time;
		var fps = 1000 * this.actual_frame_count / (ms + 1);
		var sps = 1000 * this.actual_step_count / (ms + 1);
		var text = fps.toFixed(2) + " / " + this.fps.toFixed(2) + " fps, ";
		text += sps.toFixed(2) + " / " + this.sps.toFixed(2) + " sps";

		this.ctx_unmap();
		this.ctx.font = "10pt Arial";
		this.ctx.textAlign = "left";
		this.ctx.textBaseline = "top";

		this.ctx.fillStyle = "rgb(0,0,0)";
		var textsize = this.ctx.measureText(text);
		this.ctx.fillRect(5, 5, textsize.width, 18);

		this.ctx.fillStyle = "rgb(200,200,200)";
		this.ctx.fillText(text, 5, 5);
	};

	this.draw_energy = function()
	{
		if (!this.show_energy)
			return;

		var energy = 0;

		for (var i = 0; i < this.masses.length; i++) {
			var m = this.masses[i];
			var v = Math.sqrt(m.vx*m.vx + m.vy*m.vy);
			energy += m.m * v * v * 0.5;
		}

		for (var i = 0; i < this.springs.length; i++) {
			var s = this.springs[i];
			var m1 = this.masses[s.m1], m2 = this.masses[s.m2];
			var dx = m1.x - m2.x, dy = m1.y - m2.y;
			var d = Math.sqrt(dx*dx + dy*dy) - s.s;
			energy += s.k * d * d * 0.5;
		}
		
		var text = "Total energy: " + energy.toFixed(2);

		this.ctx_unmap();
		this.ctx.font = "10pt Arial";
		this.ctx.textAlign = "left";
		this.ctx.textBaseline = "bottom";

		this.ctx.fillStyle = "rgb(0,0,0)";
		var textsize = this.ctx.measureText(text);
		this.ctx.fillRect(5, this.h-5-18, textsize.width, 18);

		this.ctx.fillStyle = "rgb(200,200,200)";
		this.ctx.fillText(text, 5, this.h-5);
	};

	this.draw_play = function()
	{
		var d = Math.sqrt(this.w*this.w + this.h*this.h) * 0.1;
		this.ctx_unmap();
		this.ctx.lineWidth = 10;
		this.ctx.strokeStyle = "rgba(200,200,200,0.5)";
		this.ctx.fillStyle = "rgba(200,200,200,0.5)";
		this.ctx.beginPath();
		this.ctx.moveTo(this.w*0.5 + d*Math.cos(0*2*Math.PI/3), this.h*0.5 + d*Math.sin(0*2*Math.PI/3));
		this.ctx.lineTo(this.w*0.5 + d*Math.cos(1*2*Math.PI/3), this.h*0.5 + d*Math.sin(1*2*Math.PI/3));
		this.ctx.lineTo(this.w*0.5 + d*Math.cos(2*2*Math.PI/3), this.h*0.5 + d*Math.sin(2*2*Math.PI/3));
		this.ctx.fill();
		this.ctx.beginPath();
		this.ctx.moveTo(this.w*0.5 + 1.5*d, this.h*0.5);
		this.ctx.arc(this.w*0.5, this.h*0.5, 1.5*d, 0, 2*Math.PI);
		this.ctx.stroke();
	}

	this.draw = function(erase_mode)
	{
		var traceWidth = Math.sqrt((this.x_max-this.x_min)*(this.x_max-this.x_min)+(this.y_max-this.y_min)*(this.y_max-this.y_min)) / Math.sqrt(this.w*this.w+this.h*this.h);

		if (erase_mode && this.trace_mode == 0) {
			this.clear_screen();
			return;
		}

		if (erase_mode) {
			this.ctx_unmap();
			if (this.old_image) {
				this.ctx.putImageData(this.old_image, 0, 0);
				// fix for bad antialiasing (chrome on android)
				this.ctx.lineWidth = 2;
				this.ctx.strokeStyle = "rgb(0,0,0)";
				this.ctx.beginPath();
				this.ctx.moveTo(0, 0);
				this.ctx.lineTo(0, this.h-1);
				this.ctx.lineTo(this.w-1, this.h-1);
				this.ctx.lineTo(this.w-1, 0);
				this.ctx.lineTo(0, 0);
				this.ctx.stroke();
			} else {
				this.clear_screen();
				if (this.grid_step > 0) {
					this.ctx_map();
					this.ctx.strokeStyle = "rgb(64,64,64)";
					this.ctx.beginPath();
					for (var x = this.x_min + ((this.x_max - this.x_min) % this.grid_step) / 2; x < this.x_max; x += this.grid_step)
						this.ctx.moveTo(x, y_min), this.ctx.lineTo(x, y_max);
					for (var y = this.y_min + ((this.y_max - this.y_min) % this.grid_step) / 2; y < this.y_max; y += this.grid_step)
						this.ctx.moveTo(x_min, y), this.ctx.lineTo(x_max, y);
					this.ctx.stroke();
				}
			}
		}

		this.ctx_map();
		if (erase_mode) {
			this.ctx.lineWidth = 2*traceWidth;
			var t = this.color_time++/100;
			var r = +Math.sin(t);
			var g = +Math.cos(t);
			var b = -Math.sin(t);
			r = r > 0 ? (50+r*155).toFixed(0) : "100";
			g = g > 0 ? (50+g*155).toFixed(0) : "100";
			b = b > 0 ? (50+b*155).toFixed(0) : "100";
			this.ctx.fillStyle = "rgba("+r+","+g+","+b+","+this.trace_alpha+")";
			this.ctx.strokeStyle = "rgba("+r+","+g+","+b+","+this.trace_alpha+")";
		} else {
			this.ctx.lineWidth = traceWidth;
			this.ctx.fillStyle = "rgb(255,255,255)";
			this.ctx.strokeStyle = "rgb(155,155,155)";
		}

		if (!erase_mode || this.trace_mode == 2)
			for (var i = 0; i < this.springs.length; i++) {
				var S = this.springs[i];
				var M1 = this.masses[S.m1];
				var M2 = this.masses[S.m2];
				this.ctx.beginPath();
				this.ctx.moveTo(M1.x, M1.y);
				this.ctx.lineTo(M2.x, M2.y);
				this.ctx.stroke();

			}

		if (!erase_mode || this.trace_mode == 1)
			for (var i = 0; i < this.masses.length; i++) {
				var M = this.masses[i];
				var r = erase_mode ? 2*traceWidth : Math.sqrt(M.m);
				this.ctx.beginPath();
				this.ctx.moveTo(M.x+r, M.y);
				this.ctx.arc(M.x, M.y, r, 0, 2*Math.PI);
				this.ctx.fill();
			}

		if (erase_mode) {
			this.ctx_unmap();
			this.old_image = this.ctx.getImageData(0, 0, this.w, this.h);
		} else {
			this.draw_fps();
			this.draw_energy();
		}
	};

	this.state_running = false;
	this.req_restart = true;
	this.req_clear = true;
	this.req_stop = false;
	this.req_delayed_stop = false;

	this.tick = function()
	{
		var date = new Date();
		var current_step = Math.round(date.valueOf() * this.sps / 1000);
		var current_frame = Math.round(date.valueOf() * this.fps / 1000);

		if (this.actual_start_time == 0) {
			this.actual_start_time = date.valueOf();
			this.actual_frame_count = 0;
			this.actual_step_count = 0;
		}

		if (this.step_counter == 0)
			this.step_counter = current_step-1;

		if (this.req_restart || this.req_clear) {
			if (this.req_restart)
				this.time = 0;
			this.clear_screen();
			this.color_time = 0;
			this.req_restart = false;
			this.req_clear = false;
		}
		else if (this.frame_counter != current_frame && !this.req_stop)
			this.draw(true);

		if (this.req_stop) {
			this.step_counter = 0;
			this.frame_counter = 0;
			this.actual_start_time = 0;
			this.actual_frame_count = 0;
			this.actual_step_count = 0;
			this.req_stop = false;
			this.state_running = false;
			return;
		}

		var steps = current_step - this.step_counter;
		if (steps > this.sps)
			steps = this.sps;
		if (steps < 0 || this.actual_step_count == 0)
			steps = 1;

		for (var i = 0; i < steps; i++)
			this.calc();
		this.actual_step_count += steps;

		if (this.frame_counter != current_frame) {
			this.actual_frame_count++;
			this.draw(false);
		}

		this.step_counter = current_step;
		this.frame_counter = current_frame;

		if (this.req_delayed_stop) {
			this.draw_play();
			this.step_counter = 0;
			this.frame_counter = 0;
			this.actual_start_time = 0;
			this.actual_frame_count = 0;
			this.actual_step_count = 0;
			this.req_delayed_stop = false;
			this.state_running = false;
			return;
		}

		this.state_running = true;
		window.setTimeout(function(that) { that.tick() }, 100/this.fps, this);
	};

	this.stop = function()
	{
		if (this.state_running)
			this.req_stop = true;
	};

	this.start = function()
	{
		this.req_stop = false;
		if (!this.state_running)
			this.tick();
	};

	this.startstop = function()
	{
		if (this.state_running)
			this.req_stop = true;
		else
			this.start();
	}

	this.init = function()
	{
		this.req_restart = true;
		this.req_delayed_stop = true;
		if (!this.state_running)
			this.tick();
	};

	this.restart = function()
	{
		this.req_restart = true;
		this.start();
	};

	this.clear = function()
	{
		if (!this.state_running) {
			this.clear_screen();
			this.draw(false);
		}
		this.req_clear = true;
	};

	this.speed = function(sps_factor, fps_factor)
	{
		this.sps *= sps_factor;
		this.fps *= fps_factor ? fps_factor : sps_factor;
		this.actual_start_time = 0;
		this.actual_frame_count = 0;
		this.actual_step_count = 0;
		this.draw_fps();
	};

	this.trace = function(mode)
	{
		this.trace_mode = mode;
		this.actual_start_time = 0;
		this.actual_frame_count = 0;
		this.actual_step_count = 0;
		this.clear_screen();
		this.draw(false);
	};

	this.grid = function(step)
	{
		this.grid_step = step;
		this.clear_screen();
		this.draw(false);
	};
}

