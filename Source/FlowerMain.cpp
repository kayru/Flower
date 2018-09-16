#include <Rush/GfxDevice.h>
#include <Rush/GfxPrimitiveBatch.h>
#include <Rush/GfxRef.h>
#include <Rush/Platform.h>
#include <Rush/Window.h>
#include <Rush/UtilRandom.h>
#include <Rush/UtilTimer.h>

struct VectorField
{
	static constexpr u32 width  = 512;
	static constexpr u32 height = 512;
	static constexpr u32 count  = width * height;

	Vec2 data[count];
};

static void initVectorField(VectorField& vf, const Vec2& value)
{
	for (Vec2& it : vf.data)
	{
		it = value;
	}
}

static const Vec2& sample(const VectorField& vf, const Vec2& uv)
{
	u32 ix = (u32)(uv.x*vf.width) % vf.width;
	u32 iy = (u32)(uv.y*vf.height) % vf.height;
	return vf.data[ix + iy * vf.width];
}

static void dampen(VectorField& vf, const Vec2& brushPos, float brushRadius)
{
	for (u32 y = 0; y < vf.height; ++y)
	{
		for (u32 x = 0; x < vf.width; ++x)
		{
			Vec2 p = Vec2((float)x, (float)y) / Vec2((float)vf.width, (float)vf.height);
			Vec2 delta = (p - brushPos);
			Vec2 absDelta(fabs(delta.x), fabs(delta.y));
			if (absDelta.x <= brushRadius && absDelta.y <= brushRadius)
			{
				Vec2& v = vf.data[x + y * vf.width];
				Vec2 forceDir = absDelta / brushRadius;
				float forceLen = min(1.0f, forceDir.length());
				v = lerp(v * 0.8f, v, forceLen);
			}
		}
	}
}

static void comb(VectorField& vf, const Vec2& brushPrev, const Vec2& brushCur, float brushRadius)
{
	Vec2 stroke = brushCur - brushPrev;
	float strokeLength = stroke.length();

	const float strokeThreshold = 0.0001f;
	if (strokeLength <= strokeThreshold) return;

	const float strokeWeight = pow(strokeLength, 1.8f);

	Vec2 strokeDir = stroke / strokeLength;

	for (u32 y = 0; y < vf.height; ++y)
	{
		for (u32 x = 0; x < vf.width; ++x)
		{
			Vec2 p = Vec2((float)x, (float)y) / Vec2((float)vf.width, (float)vf.height);
			Vec2 delta = (p - brushCur);
			Vec2 absDelta(fabs(delta.x), fabs(delta.y));

			if (absDelta.x <= brushRadius && absDelta.y <= brushRadius)
			{
				Vec2& v = vf.data[x + y * vf.width];

				Vec2 forceDir = absDelta / brushRadius;

				float forceLen = min(1.0f, forceDir.length());

				float combWeight = (1.0f - forceLen) * strokeWeight * (150.0f / (4.0f*brushRadius));

				v += strokeDir * combWeight;

				float vLength = v.length();
				if (vLength > 1.0f)
				{
					v /= vLength;
				}
			}
		}
	}
}

struct Particles
{
	static constexpr u32 count = 150000;

	Vec2 pos[count];
	Vec2 vel[count];
	u32  life[count];
};

static void initParticles(Particles& p, Rand& rng)
{
	for (u32 i = 0; i < p.count; ++i)
	{
		p.pos[i] = Vec2(rng.getFloat(0, 1), rng.getFloat(0, 1));
		p.vel[i] = 0.0f;
		p.life[i] = 0;
	}
}

static void updateParticles(Particles& p, const VectorField& vf, Rand& rng)
{
	float forceScale = 0.002f;
	float friction = 0.1f;
	for (u32 i = 0; i < p.count; ++i)
	{
		Vec2 force = 0.0f;

		if (p.pos[i].x > 0 && p.pos[i].x < 1 &&
			p.pos[i].y > 0 && p.pos[i].y < 1)
		{
			force = sample(vf, p.pos[i]) * forceScale;
		}

		p.pos[i] += p.vel[i];

		p.vel[i] *= friction;
		p.vel[i] += force;

		if (p.life[i] == 0)
		{
			p.pos[i] = Vec2(rng.getFloat(0, 1), rng.getFloat(0, 1));
			p.vel[i] = sample(vf, p.pos[i]) * forceScale;
			p.life[i] += rng.getUint(0, 80);
		}
		else
		{
			--p.life[i];
		}
	}
}

static void drawBrush(PrimitiveBatch* prim, Vec2 brushPos, float brushRadius)
{
	const u32 divisionCount = 60;
	Vec2 prev = brushPos + Vec2(brushRadius, 0);
	for (u32 i = 1; i <= divisionCount; ++i)
	{
		float t = ((float)i / (float)divisionCount);

		float st = sinf(t * TwoPi);
		float ct = cosf(t * TwoPi);

		Vec2 next = brushPos + Vec2(ct, st) * brushRadius;
		prim->drawLine(Line2(prev, next), ColorRGBA8::White());

		prev = next;
	}
}

struct State
{
	Timer timer;
	Rand rng;
	VectorField vectorField;
	Particles particles;
	PrimitiveBatch* primitiveBatch = nullptr;

	Vec2 visualDimensions = Vec2(1.0f);

	Vec2 brushPos = Vec2(0.5f);
	Vec2 brushPosPrev = brushPos;
	float brushRadius = 0.1f;

	int mouseWheel = 0;
	int mouseWheelPrev = 0;

	GfxBlendStateRef blendLerp;
	GfxBlendStateRef blendAdd;

	bool showParticles = true;
	bool showField = false;
	bool showBrush = true;

	u64 lastMouseActivityTime = 0;
};

static void startup(State* state)
{
	state->primitiveBatch = new PrimitiveBatch();

	state->blendLerp.takeover(Gfx_CreateBlendState(GfxBlendStateDesc::makeLerp()));

	GfxBlendStateDesc additiveDesc = GfxBlendStateDesc::makeAdditive();
	additiveDesc.src = GfxBlendParam::SrcAlpha;
	state->blendAdd.takeover(Gfx_CreateBlendState(additiveDesc));

	initVectorField(state->vectorField, Vec2(0.0f));
	initParticles(state->particles, state->rng);
}

static void shutdown(State* state)
{
	delete state->primitiveBatch;
	delete state;
}

static ColorRGBA hsvToRgb(float h, float s, float v)
{
	static constexpr float smallNumber = 0.00001f;

	if (s < smallNumber)
	{
		return ColorRGBA(v, v, v);
	}

	h /= 60.0f;

	int i = int(floor(h));
	float f = h - i;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));

	float r, g, b;
	switch (i) 
	{
	case 0:  r = v; g = t; b = p; break;
	case 1:  r = q; g = v; b = p; break;
	case 2:  r = p; g = v; b = t; break;
	case 3:  r = p; g = q; b = v; break;
	case 4:  r = t; g = p; b = v; break;
	default: r = v; g = p; b = q; break;
	}

	return ColorRGBA(r, g, b);
}

static ColorRGBA8 dirToColor(Vec2 dir, float saturation = 1.0f, float brightness = 1.0f)
{
	float at = 360.0f * atan2(dir.x, dir.y) / Pi;
	return hsvToRgb(at, saturation, brightness);
}

static void drawParticles(PrimitiveBatch* prim, const Particles& particles, Vec2 visualDimensions)
{
	const u32 particleCount = particles.count;
	const u32 maxVerticesPerBatch = prim->getMaxBatchVertices();
	const u32 particlesPerBatch = maxVerticesPerBatch / 2;
	const u32 batchCount = divUp(particleCount, particlesPerBatch);

	for (u32 batchId = 0; batchId < batchCount; ++batchId)
	{
		const u32 firstIndex = batchId * particlesPerBatch;
		const u32 lastIndex  = min(firstIndex + particlesPerBatch, particles.count);
		const u32 batchParticleCount = lastIndex - firstIndex;
		const u32 batchVertexCount = batchParticleCount * 2;

		PrimitiveBatch::BatchVertex* vertices = prim->drawVertices(GfxPrimitive::LineList, batchVertexCount);

		for (u32 i = 0; i < batchParticleCount; ++i)
		{
			u32 particleId = firstIndex + i;
			Vec2 pos = particles.pos[particleId] * visualDimensions;
			Vec2 dir = particles.vel[particleId] * visualDimensions;

			dir *= 6.0f;

			if (fabs(dir.x) < 1.0f && fabs(dir.y) <= 1.0f) dir.y = -1.0f;

			Line2 line(pos, pos - dir);

			ColorRGBA8 color = dirToColor(normalize(dir), 0.2f, 0.3f);

			ColorRGBA8 colorStart = color; colorStart.a = 115;
			ColorRGBA8 colorEnd = color; colorEnd.a = 0;

			vertices[i * 2 + 0].pos = Vec3(line.start.x, line.start.y, 0.0f);
			vertices[i * 2 + 0].tex = Vec2(0.0f);
			vertices[i * 2 + 0].col = colorStart;

			vertices[i * 2 + 1].pos = Vec3(line.end.x, line.end.y, 0.0f);
			vertices[i * 2 + 1].tex = Vec2(0.0f);
			vertices[i * 2 + 1].col = colorEnd;
		}
	}
}

static void drawField(PrimitiveBatch* prim, const VectorField& vf, const Vec2 visualDimensions)
{
	Vec2 fieldDimensions = Vec2(float(vf.width), float(vf.height));

	Vec2 cellSize = visualDimensions / fieldDimensions;
	Vec2 cellHalfSize = cellSize * 0.5f;

	for (u32 y = 0; y < vf.height; ++y)
	{
		for (u32 x = 0; x < vf.width; ++x)
		{
			Vec2 dir = vf.data[x + vf.width * y];
			Vec2 pos = cellHalfSize + cellSize * Vec2((float)x, (float)y);

			float dirLength = dir.length();

			ColorRGBA8 color = dirToColor(dir / dirLength, dirLength * 0.9f, min(1.0f, dirLength*5.0f));

			dirLength = min(2.0f, dirLength*20.0f);

			Line2 line(pos, pos + normalize(dir) * dirLength * cellSize);

			ColorRGBA8 colorStart = color; colorStart.a = 100;
			ColorRGBA8 colorEnd = color; colorEnd.a = 0;

			prim->drawLine(line, colorStart, colorEnd);
		}
	}
}

static void draw(State* state)
{
	Window* window = Platform_GetWindow();
	GfxContext* ctx = Platform_GetGfxContext();
	PrimitiveBatch* prim = state->primitiveBatch;

	GfxPassDesc passDesc;
	passDesc.flags = GfxPassFlags::ClearAll;
	passDesc.clearColors[0] = ColorRGBA8::Black();
	Gfx_BeginPass(ctx, passDesc);
	
	prim->begin2D(state->visualDimensions);

	if (state->showParticles) 
	{
		Gfx_SetBlendState(ctx, state->blendAdd);
		drawParticles(prim, state->particles, state->visualDimensions);
		prim->flush();
	}

	if (state->showField) 
	{
		Gfx_SetBlendState(ctx, state->blendLerp);
		drawField(prim, state->vectorField, state->visualDimensions);
		prim->flush();
	}

	if (state->showBrush)
	{
		Gfx_SetBlendState(ctx, state->blendLerp);
		drawBrush(prim, state->brushPos * state->visualDimensions, state->brushRadius * state->visualDimensions.x);
		prim->flush();
	}

	prim->end2D();

	Gfx_EndPass(ctx);
}

static void update(State* state)
{
	Window* window = Platform_GetWindow();
	const MouseState& ms = window->getMouseState();

	state->visualDimensions = window->getSizeFloat();

	state->brushPosPrev = state->brushPos;
	state->brushPos = ms.pos / state->visualDimensions;

	state->mouseWheelPrev = state->mouseWheel;
	state->mouseWheel = ms.wheelV;

	const int mouseWheelDelta = state->mouseWheel - state->mouseWheelPrev;

	const bool mouseMoved = state->brushPos != state->brushPosPrev || mouseWheelDelta != 0;

	float brushRadiusDelta = 0.0001f * float(mouseWheelDelta);
	if (mouseWheelDelta != 0)
	{
		state->brushRadius += brushRadiusDelta;
		state->brushRadius = clamp(state->brushRadius, 0.01f, 0.5f);
	}

	if (mouseMoved || ms.buttons[0] || ms.buttons[1])
	{
		state->lastMouseActivityTime = state->timer.microTime();
	}

	const u64 timeSinceLastMouseMove = state->timer.microTime() - state->lastMouseActivityTime;
	state->showBrush = timeSinceLastMouseMove < 1000000;

	if (ms.buttons[0] && mouseMoved)
	{
		comb(state->vectorField, state->brushPosPrev, state->brushPos, state->brushRadius);
	}
	else if (ms.buttons[1])
	{
		dampen(state->vectorField, state->brushPos, state->brushRadius);
	}

	updateParticles(state->particles, state->vectorField, state->rng);

	draw(state);
}

int main()
{
	AppConfig cfg;

	State* state = new State;

	cfg.onStartup  = (PlatformCallback_Startup)startup;
	cfg.onShutdown = (PlatformCallback_Shutdown)shutdown;
	cfg.onUpdate   = (PlatformCallback_Update)update;
	cfg.userData   = state;

	cfg.width  = 1024;
	cfg.height = 1024;

	cfg.name = "Flower";

#ifdef RUSH_DEBUG
	cfg.debug = true;
#endif

	return Platform_Main(cfg);
}
