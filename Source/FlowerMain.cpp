#include <Rush/GfxDevice.h>
#include <Rush/Platform.h>
#include <Rush/Rush.h>

static void update(void* userData)
{
	GfxContext* gfxContext = Platform_GetGfxContext();

	GfxPassDesc passDesc;
	passDesc.flags          = GfxPassFlags::ClearAll;
	passDesc.clearColors[0] = ColorRGBA8(11, 22, 33);
	Gfx_BeginPass(gfxContext, passDesc);

	Gfx_EndPass(gfxContext);
}

int main()
{
	AppConfig cfg;

	cfg.onStartup  = [](void*) { printf("startup\n"); };
	cfg.onShutdown = [](void*) { printf("shutdown\n"); };
	cfg.onUpdate   = update;

	cfg.name = "Flower";

#ifdef RUSH_DEBUG
	cfg.debug = true;
#endif

	return Platform_Main(cfg);
}
