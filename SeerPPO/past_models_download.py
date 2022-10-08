import aiohttp
import asyncio
import glob
from tqdm import tqdm
from multiprocessing import Queue, Process
import os
import time


async def download_models(session, url, disable=False):
    local_files = [os.path.basename(x) for x in glob.glob("./Models/*.pt")]
    async with session.get(url + "/pastmodels") as resp:
        assert resp.status == 200
        files = await resp.read()
        files = files.decode('ascii').split(",")
        file_to_get = set(files) - set(local_files)
        files_to_delete = set(local_files) - set(files)

        for f in files_to_delete:
            if f == '':
                continue
            os.remove("./Models/{f}".format(f=f))

        for f in tqdm(file_to_get, disable=disable):
            if f == '':
                continue
            async with session.get(url + "/Models/{f}".format(f=f)) as resp2:
                assert resp2.status == 200
                with open("./Models/{f}".format(f=f), "wb") as newfile:
                    newfile.write(await resp2.read())


async def past_model_downloader_async(url):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0, connect=60, sock_connect=60, sock_read=60)) as session:
        while True:
            time.sleep(30)
            await download_models(session, url, disable=True)


def past_model_downloader(url):
    asyncio.run(past_model_downloader_async(url))


def start_past_model_downloader(url):
    p = Process(target=past_model_downloader, args=(url,))
    p.start()
