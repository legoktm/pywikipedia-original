# -*- coding: utf-8  -*-

__version__ = '$Id$'

import family

# The wikispecies family

class Family(family.Family):
    def __init__(self):
        family.Family.__init__(self)
        self.name = 'species'
        self.langs = {
            'species': 'species.wikimedia.org',
        }

        self.namespaces[4] = {
            '_default': [u'Wikispecies', self.namespaces[4]['_default']],
        }
        self.namespaces[5] = {
            '_default': [u'Wikispecies talk', self.namespaces[5]['_default']],
        }

        self.interwiki_forward = 'wikipedia'
        self.cross_projects = [
            'wikipedia', 'wiktionary', 'wikibooks', 'wikiquote', 'wikisource',
            'wikinews', 'wikiversity', 'meta', 'mediawiki', 'test', 'incubator',
            'commons',
        ]

    def shared_image_repository(self, code):
        return ('commons', 'commons')

    if family.config.SSL_connection:

        def protocol(self, code):
            return 'https'