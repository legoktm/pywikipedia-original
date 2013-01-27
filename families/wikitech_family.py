# -*- coding: utf-8  -*-

__version__ = '$Id$'

import family

# The Wikitech family

class Family(family.Family):
    def __init__(self):
        super(Family, self).__init__()
        self.name = 'wikitech'
        self.langs = {
            'en': 'wikitech.wikimedia.org',
        }

        self.namespaces[4] = {
            '_default': [u'Wikitech', self.namespaces[4]['_default']],
        }
        self.namespaces[5] = {
            '_default': [u'Wikitech talk', self.namespaces[5]['_default']],
        }

    def version(self, code):
        return '1.19wmf2'

    def scriptpath(self, code):
        return ''
