from tastytrade import DXLinkStreamer
from tastytrade.dxfeed import EventType
from tastytrade import ProductionSession

password = "Alsinas:2440"
username = "verave.javu@gmail.com"
session = ProductionSession(username, password)


async with DXLinkStreamer(session) as streamer:
    await streamer.subscribe(EventType.QUOTE, [sym])
    quotes = {}
    async for quote in streamer.listen(EventType.QUOTE):
        quotes[quote.eventSymbol] = quote
        break

    print(quotes)
