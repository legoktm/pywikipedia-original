# -*- coding: utf-8 -*-

__version__ = '$Id$'

# The new wikivoyage family that is hosted at wikimedia

import family

class Family(family.WikimediaFamily):
    def __init__(self):
        super(Family, self).__init__()
        self.name = 'wikivoyage'
        self.languages_by_size = [
            'en', 'de', 'pt', 'nl', 'fr', 'it', 'ru', 'sv', 'es', 'ro', 'pl',
        ]

        self.langs = dict([(lang, '%s.wikivoyage.org' % lang)
                           for lang in self.languages_by_size])

        # Override defaults
        self.namespaces[2]['ru'] = [u'Участник', u'Участница']
        self.namespaces[3]['fr'] = [u'Discussion utilisateur', u'Discussion Utilisateur']
        self.namespaces[2]['fr'] = [u'Utilisateur']
        self.namespaces[12]['nl'] = [u'Help']
        self.namespaces[3]['pt'] = [u'Utilizador Discussão', u'Usuário Discussão', u'Utilizadora Discussão']
        self.namespaces[2]['pt'] = [u'Utilizador', u'Usuário', u'Utilizadora']
        self.namespaces[9]['ro'] = [u'Discuție MediaWiki', u'Discuţie MediaWiki']
        self.namespaces[3]['pl'] = [u'Dyskusja użytkownika', u'Dyskusja użytkowniczki']
        self.namespaces[2]['pl'] = [u'Użytkownik', u'Użytkowniczka']

        # Most namespaces are inherited from family.Family.
        # Translation used on all wikis for the different namespaces.
        # (Please sort languages alphabetically)
        # You only need to enter translations that differ from _default.
        self.namespaces[4] = {
            'de': u'Wikivoyage',
            'en': u'Wikivoyage',
            'es': u'Wikiviajes',
            'fr': u'Wikivoyage',
            'it': u'Wikivoyage',
            'nl': u'Wikivoyage',
            'pl': u'Wikipodróże',
            'pt': u'Wikivoyage',
            'ro': u'Wikivoyage',
            'ru': u'Wikivoyage',
            'sv': u'Wikivoyage',
        }

        self.namespaces[5] = {
            'de': u'Wikivoyage Diskussion',
            'en': u'Wikivoyage talk',
            'es': u'Wikiviajes discusión',
            'fr': u'Discussion Wikivoyage',
            'it': u'Discussioni Wikivoyage',
            'nl': u'Overleg Wikivoyage',
            'pl': u'Dyskusja Wikipodróży',
            'pt': u'Wikivoyage Discussão',
            'ro': u'Discuție Wikivoyage',
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

        # Global bot allowed languages on http://meta.wikimedia.org/wiki/Bot_policy/Implementation#Current_implementation
        self.cross_allowed = ['es', 'ru', ]
