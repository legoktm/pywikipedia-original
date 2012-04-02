# -*- coding: utf-8 -*-
"""
This script will collect articles (currently only from main namespace) that
have n dash or m dash character in their title, and create a redirect to them
from the corresponding hyphenated title.

TODO:
- prompting for other namespaces and start
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

site = pywikibot.getSite()
redirword = site.redirect()
regex = ur'.*[–—]' # Alt 0150 (n dash), alt 0151 (m dash), respectively.
gen = RPG(site.allpages(namespace=0, includeredirects=False), [regex])
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
