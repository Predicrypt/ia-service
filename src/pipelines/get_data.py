from kfp.components import create_component_from_func, OutputPath
from kfp.compiler import Compiler
import os


def get_all_data(symbol: str, interval: str, limit: int, output_path: OutputPath(str)):

    import asyncio
    from aiohttp import ClientSession
    import pickle

    intervalDict = {
        "1h": 3600*1000,
        "4h": 4*3600*1000,
        "8h": 8*3600*1000
    }

    klines = []

    async def getKlines(session, url: str, params: dict):
        async with session.get(url, params=params) as res:
            klines = await res.json()
            return klines

    async def getTime(session):
        url = "https://api.binance.com/api/v3/time"
        async with session.get(url) as res:
            return await res.json()

    def roundTime(time: int):
        BASE = 100000

        r = time % BASE

        return time + BASE - r

    async def getData(symbol: str, interval: str, limit: int):


        import os

        MAX_DATA_REQUEST = 1000
        URL = "https://api.binance.com/api/v3/klines"

        async with ClientSession() as session:
            time_started = (await getTime(session))["serverTime"]
            tasks = []
            for i in range(0, int(limit/MAX_DATA_REQUEST)):
                end_time = int(time_started - i*(MAX_DATA_REQUEST)
                               * intervalDict[interval])
                end_time = roundTime(end_time)
                tasks.append(asyncio.ensure_future(getKlines(session, URL, {
                    "interval": interval, "symbol": symbol, "limit": MAX_DATA_REQUEST, "endTime": end_time})))

            data = await asyncio.gather(*tasks)

            for req in data:
                for kline in req:
                    klines.insert(0, kline)

    asyncio.run(getData(symbol, interval, limit))
    with open(output_path, 'wb') as fp:
        pickle.dump(klines, fp)



get_all_data_component = create_component_from_func(
    func=get_all_data,
    output_component_file=f'{os.getcwd()}/ia-service/src/pipelines/definitions/GetAllDataComponent.yaml',
    base_image='python:3.8',
    packages_to_install=['aiohttp===3.7.4', 'asyncio']
)

if __name__ == '__main__':
    get_all_data('ETHUSDT', '1h', 3000)
