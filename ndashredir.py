# -*- coding: utf-8 -*-
"""
This script will collect articles that have n dash or m dash character in their
title, and create a redirect to them automatically from the corresponding
hyphenated title. If the target exists, will be skipped.
It may take several hours. You may quit by Ctrl C at any time and continue
later. Type the first few characters of the last shown title after -start.

The script is primarily designed for work in article namespace, but can be used
in any other one. Use in accordance with the rules of your community.

Known parameters:
-start            Will start from the given title (it does not have to exist).
                  Parameter may be given as "-start" or "-start:tile"
                  Defaults to '!'.
-namespace        Works in the given namespace (only one at a time). Parameter
-ns               may be given as "-ns:<number>" or "-namespace:<number>".
                  Defaults to 0 (main namespace).

"""
"""
TODO:
- listing existing hyphenated titles to a file/wikipage instead of just skipping
"""
#
# (C) Bináris, 2012
#
# Distributed under the terms of the MIT license.
#
__version__='$Id$'

import wikipedia as pywikibot
from pagegenerators import RegexFilterPageGenerator as RPG
from pywikibot import i18n

def main(*args):
    regex = ur'.*[–—]' # Alt 0150 (n dash), alt 0151 (m dash), respectively.
    ns = 0
    start = '!'

    # Handling parameters:
    for arg in pywikibot.handleArgs(*args):
        pass
        if arg == '-start':
            start = pywikibot.input(
                    u'From which title do you want to continue?')
        elif arg.startswith('-start:'):
            start = arg[7:]
        elif arg in ['-ns', '-namespace']:
            ns = pywikibot.input(u'Which namespace should we process?')
        elif arg.startswith('-ns:') or arg.startswith('-namespace:'):
            ns = arg[arg.find(':')+1:]
    site = pywikibot.getSite()
    redirword = site.redirect()
    gen = RPG(site.allpages(
          start=start, namespace=ns, includeredirects=False), [regex])

    # Processing:
    for page in gen:
        title = page.title()
        editSummary = i18n.twtranslate(site, 'ndashredir-create',
                                           {'title': title})
        newtitle = title.replace(u'–','-').replace(u'—','-')
        # n dash -> hyphen, m dash -> hyphen, respectively
        redirpage = pywikibot.Page(site, newtitle)
        if redirpage.exists():
            if redirpage.isRedirectPage() and \
                        redirpage.getRedirectTarget() == page:
                pywikibot.output(
                    u'[[%s]] already redirects to [[%s]], nothing to do with it.'
                    % (newtitle, title))
            else:
                pywikibot.output(
                    (u'Skipping [[%s]] beacuse it exists already with a ' +
                    u'different content.') % newtitle)
                # TODO: list it for further examination to a file or wikipage
        else:
            text = u'#%s[[%s]]' % (redirword, title)
            redirpage.put(text, editSummary)
        # Todo: output the title upon Ctrl C? (KeyboardInterrupt always hits
        # RegexFilterPageGenerator and cannot be catched in this loop.)

if __name__ == "__main__":
    try:
        main()
    finally:
        pywikibot.stopme()
