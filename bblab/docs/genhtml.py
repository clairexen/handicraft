#!/usr/bin/env python

import sys

idCount = 0

def splitThisSegment(txt):
    pos = 0
    depth = 1
    if txt[0] == '-':
        while txt[depth] == '-':
            depth = depth + 1
            pos = pos + 1
        while pos+depth < len(txt) and txt[pos:pos+depth] != txt[0:depth]:
            pos = pos + 1
        return (txt[depth:pos], txt[pos+depth+1:])
    while depth > 0 and pos < len(txt):
        if txt[pos] == '<':
            depth = depth + 1
        if txt[pos] == '>':
            depth = depth - 1
        pos = pos + 1
    return (txt[0:pos-1], txt[pos+1:])

def genHtml_generic(txt, main_mode):
    body = ""
    state = -1
    while txt != "":
        if txt[0] == '<':
            (sub, txt) = splitThisSegment(txt[1:])
            (sub_type, sub_body) = sub.split(":", 1)
            body = body + tag_handlers[sub_type](sub_type, sub_body)
        elif txt[0:2] == "[[":
            p = txt.find("]]")
            keyword = txt[2:p]
            txt = txt[p+2:]
            p = keyword.find('|')
            if p == -1:
                caption = keyword
            else:
                caption = keyword[p+1:]
                keyword = keyword[0:p]
            if keyword2url.has_key(keyword):
                body = body + '<a href="{1}">{0}</a>'.format(caption, keyword2url[keyword])
            else:
                body = body + '<a href="{1}">{0}</a>'.format(caption, keyword)
        elif main_mode:
            if txt[0].strip() != "":
                state = 0
            elif state == 0 and txt[0] == "\n":
                state = 1
            elif state == 1 and txt[0] == "\n":
                body = body + "<p/>"
                state = -1
            body = body + txt[0]
            txt = txt[1:]
        else:
            body = body + txt[0]
            txt = txt[1:]
    return body

def genHtml(txt):
    return genHtml_generic(txt, False)

def genHtml_main(txt):
    return genHtml_generic(txt, True)

def genHtml_title(tagname, txt):
    return "<h1>" + genHtml(txt) + "</h1>"

def genHtml_h1(tagname, txt):
    return "<h3>" + genHtml(txt) + "</h3>"

def genHtml_h2(tagname, txt):
    return "<h4>" + genHtml(txt) + "</h4>"

def genHtml_b(tagname, txt):
    return "<strong>" + genHtml(txt) + "</strong>"

def genHtml_gallery(tagname, txt):
    global idCount
    imgList = [ ]
    for img in txt.split("\n"):
        img = img.strip()
        if img != "":
            (img_file, img_desc) = img.split(" ", 1)
            imgList.append( (idCount, img_file, genHtml(img_desc)) )
            idCount = idCount + 1
    body = """
        <p align="center"><table><tr><td>
    """
    for img in imgList:
        displayDecl = ""
        if img[0] != imgList[0][0]:
            displayDecl = "display:none;"
        body = body + """
            <div id="pic{0}" align="center" style="{3}"><a href="{1}" target="_blank"><img src="{1}" style="max-width: 600px; border:0"/></a><p/>{2}</div>
        """.format(img[0], img[1], img[2], displayDecl)
    body = body + """
        </td><td valign="top">
    """
    for img in imgList:
        onclickAction = ""
        for i in imgList:
            if i[0] == img[0]:
                onclickAction = onclickAction + "document.getElementById('pic{0}').setAttribute('style', '');".format(i[0])
                onclickAction = onclickAction + "document.getElementById('pre{0}').setAttribute('style', 'width: 100px; padding:5px; margin:5px; background:gray;');".format(i[0])
            else:
                onclickAction = onclickAction + "document.getElementById('pic{0}').setAttribute('style', 'display:none;');".format(i[0])
                onclickAction = onclickAction + "document.getElementById('pre{0}').setAttribute('style', 'width: 100px; padding:5px; margin:5px;');".format(i[0])
        bgDecl = ""
        if img[0] == imgList[0][0]:
            bgDecl = "background:gray"
        body = body + """
            <img id="pre{0}" src="{1}" style="width: 100px; padding:5px; margin:5px; {2}" onclick="{3}"/><br/>
        """.format(img[0], img[1], bgDecl, onclickAction)
    body = body + """
        </td></tr></table></p>
    """
    return body

def genHtml_code(tagname, txt):
    return """
        <pre align="center" style="margin-left:5em; margin-right:5em; padding:1em; background-color:lightgray; border:1px dashed black;">{0}</pre>
    """.format(txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").strip().replace("\t", "    "))

tag_handlers = {
    "title": genHtml_title,
    "h1": genHtml_h1,
    "h2": genHtml_h2,
    "b": genHtml_b,
    "gallery": genHtml_gallery,
    "code": genHtml_code
}

keyword2url = {
    "V-USB": "http://www.obdev.at/products/vusb/index.html",
    "Arduino": "http://www.arduino.cc/"
}

f_in = open(sys.argv[1], "r")
f_out = open(sys.argv[2], "w")

f_out.write(genHtml_main(f_in.read()))

f_in.close()
f_out.close()

