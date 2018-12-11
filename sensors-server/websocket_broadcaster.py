#!/usr/bin/env python3

import asyncio
import collections
import json
import logging
import ssl
import time

import websockets.server

logger = logging.getLogger(__name__)


class WebsocketBroadcaster:
    '''
    function test(seconds){
        function make_callback(i) {
            return function(event) {
                console.log('' + i + ': ' + event.data);
            }
        }
        let s1 = new WebSocket("ws://localhost:8090");
        let s2 = new WebSocket("ws://localhost:8090");
        s1.onmessage = make_callback(1);
        s2.onmessage = make_callback(2);
        setTimeout(function(){s1.close();s2.close();}, 1000*seconds);
    }
    test(35)
    
    Console output should contain pairs like:
    1: {"message": 7, "timestamp": 1544499855.1407769}
    2: {"message": 7, "timestamp": 1544499855.1407769}
    '''
    class Payload(collections.namedtuple('Payload', ('message', 'timestamp'))):
        def json(self):
            return json.dumps(dict(self._asdict()))

    def __init__(self, certfile=None, keyfile=None, address='127.0.0.1', port=8090):
        self.certfile = certfile
        self.keyfile = keyfile
        self.address = address
        self.port = port
        self.clients = set()
        self.payload = None
        self.payload_ready = asyncio.Event()

    def broadcast(self, message):
        self.payload = WebsocketBroadcaster.Payload(message, time.time())
        self.payload_ready.set()

    def __str__(self):
        return 'WebsocketBroadcaster on {}://{}:{}'.format(
            'wss' if self.certfile and self.keyfile else 'ws',
            self.address,
            self.port
        )

    async def client_handler_coroutine(self, client, path):
        logger.info('%s: client %s connected', self, client.remote_address)
        self.clients.add(client)
        while True:
            try:
                from_client = await client.recv()
                logger.info('%s: received %s from client %s', self, client.remote_address, from_client)
            except websockets.exceptions.ConnectionClosed as e:
                logger.info('%s: client %s closed connection (%s)', self, client.remote_address, e)
                self.clients.discard(client)
                break
    
    async def broadcast_coroutine(self):
        while True:
            await self.payload_ready.wait()
            payload_json = self.payload.json()
            self.payload_ready.clear()

            if not self.clients:
                # if a tree falls in a forest but nobody hears it...
                logger.info('%s: payload %s not sent to any clients', self, payload_json)
                continue

            futures_by_client = dict([
                [client, asyncio.ensure_future(client.send(payload_json))]
                for client in self.clients
            ])

            await asyncio.wait(futures_by_client.values())

            closed_clients = set()
            delivered_clients = set()
            for client in self.clients:
                try:
                    futures_by_client.get(client).result()
                    delivered_clients.add(client)
                except websockets.exceptions.ConnectionClosed as e:
                    closed_clients.add(client)
                except Exception as e:
                    logger.warn('%s: exception sending payload to client %s (%s)', self, client.remote_address, e)
            if closed_clients:
                self.clients = self.clients.difference(closed_clients)
                logger.info('%s: clients %s closed connection before payload %s could be sent', self,
                            list((client.remote_address for client in closed_clients)), payload_json)
            if delivered_clients:
                logger.info('%s: payload %s delivered to clients %s', self, payload_json,
                            list(client.remote_address for client in delivered_clients))

    async def __aenter__(self):
        if self.certfile and self.keyfile:
            sslcontext = ssl.SSLContext()
            sslcontext.load_cert_chain(self.certfile, self.keyfile)
        else:
            sslcontext = None
        self.server = await websockets.server.serve(self.client_handler_coroutine, self.address, self.port, ssl=sslcontext)
        self.broadcast_future = asyncio.ensure_future(self.broadcast_coroutine())
        logger.info('%s: listening...', self)
        return self

    async def __aexit__(self, *args):
        self.server.close()
        self.broadcast_future.cancel()
        await self.server.wait_closed()
        logger.info('%s has been shut down', self)


async def main():
    async with WebsocketBroadcaster() as server:
        for i in range(30):
            server.broadcast(i)
            await asyncio.sleep(1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
