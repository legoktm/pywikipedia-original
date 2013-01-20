# -*- coding: utf-8  -*-
""" Script to determine the pywikipedia version (tag, revision and date) """
#
# (C) Merlijn 'valhallasw' van Deen, 2007-2008
# (C) xqt, 2010-2011
# (C) Pywikipedia bot team, 2007-2013
#
# Distributed under the terms of the MIT license.
#
__version__ = '$Id$'

from pywikibot.version import *
import config


if __name__ == '__main__':
    print 'Pywikipedia %s' % getversion()
    print 'Python %s' % sys.version
    print 'config-settings:'
    print 'use_api =', config.use_api
    print 'use_api_login =', config.use_api_login
    if not __import__('unicodedata').normalize('NFC', u'\u092e\u093e\u0930\u094d\u0915 \u091c\u093c\u0941\u0915\u0947\u0930\u092c\u0930\u094d\u0917') == u'\u092e\u093e\u0930\u094d\u0915 \u091c\u093c\u0941\u0915\u0947\u0930\u092c\u0930\u094d\u0917':
        print u'unicode test: triggers problem #3081100'
    else:
        print u'unicode test: ok'
