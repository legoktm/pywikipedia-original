# -*- coding: utf-8  -*-

__version__ = '$Id$'

import family

# The Wikimedia Incubator family

class Family(family.WikimediaFamily):
    def __init__(self):
        super(Family, self).__init__()
        self.name = 'incubator'
        self.langs = {
            'incubator': 'incubator.wikimedia.org',
        }
        self.namespaces[4] = {
            '_default': [u'Incubator', u'I', self.namespaces[4]['_default']],
        }
        self.namespaces[5] = {
            '_default': [u'Incubator talk', self.namespaces[5]['_default']],
        }
        self.namespaces[100] = {
            '_default': u'Lost',
        }
        self.namespaces[101] = {
            '_default': u'Lost talk',
        }
        self.namespaces[1198] = {
            '_default': u'Translations',
        }
        self.namespaces[1199] = {
            '_default': u'Translations talk',
        }
        self.interwiki_forward = 'wikipedia'

    if family.config.SSL_connection:

        def protocol(self, code):
            return 'https'
