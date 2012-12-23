# -*- coding: utf-8  -*-

__version__ = '$Id: wikidata_family.py 10591 2012-10-20 amir $'

import family

# The wikidata family

class Family(family.Family):
    def __init__(self):
        family.Family.__init__(self)
        self.name = 'wikidata'
        self.langs = {
            'wikidata': 'wikidata.org',
            'test':     'wikidata-test-repo.wikimedia.de',
        }

        self.namespaces[4] = {
            '_default': [u'Wikidata', u'WD', 'Project'],
        }
        self.namespaces[5] = {
            '_default': [u'Wikidata talk', u'WT', 'Project talk'],
        }
        self.namespaces[120] = {
            '_default': u'Property',
        }
        self.namespaces[121] = {
            '_default': u'Property talk',
        }
        self.namespaces[122] = {
            '_default': u'Query',
        }
        self.namespaces[123] = {
            '_default': u'Query talk',
        }
        self.namespaces[710] = {
            '_default': u'TimedText',
        }
        self.namespaces[711] = {
            '_default': u'TimedText talk',
        }
        self.namespaces[1198] = {
            '_default': u'Translations',
        }
        self.namespaces[1199] = {
            '_default': u'Translations talk',
        }        
        self.cross_projects = [
            'wikipedia', 'wiktionary', 'wikibooks', 'wikiquote', 'wikisource',
            'wikinews', 'wikiversity', 'meta', 'test', 'incubator', 'commons',
            'species', 'mediawiki'
        ]

    if family.config.SSL_connection:
        def protocol(self, code):
            return 'https'
