# -*- coding: utf-8  -*-

__version__ = '$Id: wikidata_family.py 10591 2012-10-20 amir $'

import family

# The Wikidata
# user-config.py: usernames['wikidata']['wikidata'] = 'User name'

class Family(family.Family):
    def __init__(self):
        family.Family.__init__(self)
        self.name = 'wikidata'

        self.langs = {
            'wikidata': 'wikidata-test-repo.wikimedia.de',
        }
#http://wikidata-test-repo.wikimedia.de/w/api.php?action=query&meta=siteinfo&siprop=namespaces
        self.namespaces[4] = {
            '_default': u'Wikidata-test',
        }
        self.namespaces[5] = {
            '_default': u'Wikidata-test talk',
        }
        self.namespaces[102] = {
            '_default': u'Property',
        }
        self.namespaces[103] = {
            '_default': u'Property talk',
        }
        self.namespaces[104] = {
            '_default': u'Query',
        }
        self.namespaces[105] = {
            '_default': u'Query talk',
        }
        self.cross_projects = [
            'wikipedia', 'wiktionary', 'wikibooks', 'wikiquote', 'wikisource',
            'wikinews', 'wikiversity', 'meta', 'test', 'incubator', 'commons',
            'species', 'mediawiki'
        ]
#I checked and https was not supported but i think it will
#    if family.config.SSL_connection:
#        def protocol(self, code):
#            return 'https'
