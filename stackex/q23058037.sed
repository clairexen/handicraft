# http://stackoverflow.com/questions/23058037/replace-the-text-in-sed
# sed -f q23058037.sed q23058037.txt

:loop
/<\/funcprototype>/ ! { N; b loop; }
s/\n/ /g;

s/<\/\?\(funcdef\|parameter\|function\|funcprototype\)>//g;
s/<paramdef>/(/g;
s/<\/paramdef>/)/g;
s/) *(/, /g;

s/  */ /g;
s/^ //;
s/ $//;
s/ (/(/;
