#!/usr/bin/env python3

import re
import xml.etree.ElementTree as et
import urllib.request
import email.utils

with urllib.request.urlopen('https://www.reddit.com/r/yosys/.rss') as response:
    main_rss = response.read()
main_rss_root = et.fromstring(main_rss.decode('utf-8'))

for item in main_rss_root.findall('.//item'):
    title = item.find('title').text
    link = item.find('link').text
    pubdate = item.find('pubDate').text

    match = re.match(r".*/r/yosys/comments/([^/]+)/.*/", link)
    if not match: continue
    idstr = match.group(1)

    ts = email.utils.parsedate_tz(pubdate)
    ts = email.utils.mktime_tz(ts)

    with open("%s.idx" % idstr, "w") as f:
        print("idstr %s" % idstr, file=f)
        print("title %s" % title, file=f)
        print("link %s" % link, file=f)
        print("pubdate %s %s" % (ts, pubdate), file=f)

    with urllib.request.urlopen(link + '.rss') as response:
        post_rss = response.read()
    post_rss_root = et.fromstring(post_rss.decode('utf-8'))

    print("")
    print("%s: %s" % (idstr, title))
    print(link)

