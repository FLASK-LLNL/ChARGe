import os, anyio, httpx


async def go():
    url = os.environ["LIVAI_BASE_URL"].rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    # default behavior
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers=headers)
        print("default client:", r.status_code)

    # ignore env + prefer HTTP/1.1
    async with httpx.AsyncClient(
        timeout=15, trust_env=False, verify=False, http2=False
    ) as c:
        r = await c.get(url, headers=headers)
        print("no-env/h1 client:", r.status_code)


anyio.run(go)
