# -*- coding: utf-8 -*-
"""
Add or change categories on a number of pages. Usage: catall.py name - goes
through pages, starting at 'name'. Provides the categories on the page and asks
whether to change them. If no starting name is provided, the bot starts at 'A'.

Options:
-onlynew : Only run on pages that do not yet have a category.
"""
#
# (C) Rob W.W. Hooft, Andre Engels, 2004
# (C) Pywikipedia bot team, 2004-2011
#
# Distributed under the terms of the MIT license.
#
__version__ = '$Id$'
#

import sys
import wikipedia as pywikibot
from pywikibot import i18n

def choosecats(pagetext):
    chosen=[]
    flag=False
    length=1000
    print ("Give the new categories, one per line.")
    print ("Empty line: if the first, don't change. Otherwise: Ready.")
    print ("-: I made a mistake, let me start over.")
    print ("?: Give the text of the page with GUI.")
    print ("??: Give the text of the page in console.")
    print ("xx: if the first, remove all categories and add no new.")
    print ("q: quit.")
    while flag == False:
        choice=pywikibot.input(u"?")
        if choice=="":
            flag=True
        elif choice=="-":
            chosen=choosecats(pagetext)
            flag=True
        elif choice=="?":
            import editarticle
            editor = editarticle.TextEditor()
            newtext = editor.edit(pagetext)
        elif choice =="??":
            pywikibot.output(pagetext[0:length])
            length = length+500
        elif choice=="xx" and chosen==[]:
            chosen = None
            flag=True
        elif choice=="q":
            print "quit..."
            sys.exit()
        else:
            chosen.append(choice)
    return chosen

def make_categories(page, list, site = None):
    if site is None:
        site = pywikibot.getSite()
    pllist=[]
    for p in list:
        cattitle="%s:%s" % (site.category_namespace(), p)
        pllist.append(pywikibot.Page(site,cattitle))
    page.put_async(pywikibot.replaceCategoryLinks(page.get(), pllist),
                   comment=i18n.twtranslate(site.lang, 'catall-changing'))

def main():
    docorrections=True
    start=[]

    for arg in pywikibot.handleArgs():
        if arg == '-onlynew':
            docorrections=False
        else:
            start.append(arg)

    if start == []:
        start='A'
    else:
        start=' '.join(start)

    mysite = pywikibot.getSite()

    for p in mysite.allpages(start = start):
        try:
            text=p.get()
            cats=p.categories()
            if cats == []:
                pywikibot.output(u"========== %s ==========" % p.title())
                print "No categories"
                print "-" * 40
                newcats=choosecats(text)
                if newcats != [] and newcats is not None:
                    make_categories(p, newcats, mysite)
            elif docorrections:
                pywikibot.output(u"========== %s ==========" % p.title())
                for c in cats:
                    pywikibot.output(c.title())
                print "-" * 40
                newcats=choosecats(text)
                if newcats is None:
                    make_categories(p, [], mysite)
                elif newcats != []:
                    make_categories(p, newcats, mysite)
        except pywikibot.IsRedirectPage:
            pywikibot.output(u'%s is a redirect' % p.title())

if __name__ == "__main__":
    try:
        main()
    finally:
        pywikibot.stopme()
