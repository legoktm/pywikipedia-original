# -*- coding: utf-8 -*-

__version__ = '$Id$'

# The new wikivoyage family that is hosted at wikimedia

import family

class Family(family.Family):
    def __init__(self):
        family.Family.__init__(self)
        self.name = 'wikivoyage'
        self.languages_by_size = [
            'de', 'en', 'fr', 'it', 'nl', 'ru','sv',
        ]

        self.langs = dict([(lang, '%s.wikivoyage.org' % lang) for lang in self.languages_by_size])

        self.namespaces[4] = {
            'de': u'Wikivoyage',
            'en': u'Wikivoyage',
            'fr': u'Wikivoyage',
            'it': u'Wikivoyage',
            'nl': u'Wikivoyage',
            'ru': u'Wikivoyage',
            'sv': u'Wikivoyage',
        }

        self.namespaces[5] = {
            'de': u'Wikivoyage Diskussion',
            'en': u'Wikivoyage talk',
            'fr': u'Discussion Wikivoyage',
            'it': u'Discussioni Wikivoyage',
            'nl': u'Overleg Wikivoyage',
            'ru': u'Обсуждение Wikivoyage',
            'sv': u'Wikivoyagediskussion',
        }

        self.namespaces[100] = {
            'de': u'Portal',
            'it': u'Portale',
        }

        self.namespaces[101] = {
            'de': u'Portal Diskussion',
            'it': u'Discussioni portale',
        }

        self.namespaces[102] = {
            'de': u'Wahl',
            'it': u'Elezione',
        }

        self.namespaces[103] = {
            'de': u'Wahl Diskussion',
            'it': u'Discussioni elezione',
        }

        self.namespaces[104] = {
            'de': u'Thema',
            'it': u'Tematica',
        }

        self.namespaces[105] = {
            'de': u'Thema Diskussion',
            'it': u'Discussioni tematica',
        }

        self.namespaces[106] = {
            'de': u'Nachrichten',
            'it': u'Notizie',
        }

        self.namespaces[107] = {
            'de': u'Nachrichten Diskussion',
            'it': u'Discussioni notizie',
        }

        self.cross_projects = [
            'wikipedia', 'wiktionary', 'wikibooks', 'wikiquote', 'wikisource',
            'wikinews', 'wikiversity', 'meta', 'mediawiki', 'test', 'incubator',
            'commons', 'species',
        ]

    def scriptpath(self, code):
        return u'/w'

    def shared_image_repository(self, code):
        return ('commons', 'commons')

    def shared_data_repository(self, code):
        return ('wikidata', 'wikidata')

    if family.config.SSL_connection:

        def protocol(self, code):
            return 'https'
