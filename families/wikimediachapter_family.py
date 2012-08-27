# -*- coding: utf-8 -*-


# The wikis of Chapters of the Wikimedia Foundation living at a xy.wikimedia.org url

import family

class Family(family.Family):
    def __init__(self):
        family.Family.__init__(self)
        self.name = 'wikimediachapter'
        self.countries = [ 'ar', 'bd', 'dk', 'fi', 'mk', 'mx', 'nl', 'no', 'nyc', 'pl', 'rs', 'ru', 'se', 'ua', 'uk', 've' ]
        self.countrylangs = { 'ar': 'es', 'bd': 'bn', 'dk': 'da', 'fi': 'fi', 'mk': 'mk', 'mx': 'es', 'nl': 'nl', 'no': 'no', 'nyc': 'en', 'pl': 'pl', 'rs': 'sr', 'ru': 'ru', 'se': 'sv', 'ua': 'uk', 'uk': 'en-gb', 've': 'en' }
        self.langs = dict([(country, '%s.wikimedia.org' % country) for country in self.countries])

        for country in self.countries:
            for ns in self.namespaces:
                self.namespaces[ns][country] = self.namespaces[ns][self.countrylangs[country]] if self.countrylangs[country] in self.namespaces[ns] else self.namespaces[ns]['_default']

        self.namespaces[4] = {
            '_default': 'Wikimedia',
            'bd': u'উইকিমিডিয়া বাংলাদেশ',
            'ru': u'Викимедиа',
            'rs': u'Викимедија',
            'mk': u'Викимедија',
            'ua': u'Вікімедіа'
        }

        self.namespaces[5] = {
            '_default':  self.namespaces[5]['_default'],
            'bd': u'উইকিমিডিয়া বাংলাদেশ আলোচনা',
            'dk': 'Wikimedia diskussion',
            'mk': u'Разговор за Викимедија',
            'mx': u'Wikimedia discusión',
            'nl': u'Overleg Wikimedia',
            'no': 'Wikimedia-diskusjon',
            'nyc': 'Wikimedia talk',
            'pl': 'Dyskusja Wikimedia',
            'ru': u'Обсуждение Викимедиа',
            'rs': u'Разговор о Викимедија',
            'se': u'Wikimediadiskussion',
            'ua': u'Обговорення Вікімедіа',
            'uk': u'Wikimedia talk',
            've': u'Wikimedia talk',
            'fi': u'Keskustelu Wikimediasta'
        }

        self.namespaces[9]['dk'] = u'MediaWiki diskussion'
        self.namespaces[13]['dk'] = u'Hjælp diskussion'

        self.namespaces[2]['pl'] = u'Użytkownik'
        self.namespaces[3]['pl'] = u'Dyskusja użytkownika'

        self.namespaces[90] = { 'fi': u'Viestiketju', 'se': u'Tråd' }
        self.namespaces[91] = { 'fi': u'Keskustelu viestiketjusta', 'se': u'Tråddiskussion' }
        self.namespaces[92] = { 'fi': u'Yhteenveto', 'se': u'Summering' }
        self.namespaces[93] = { 'fi': u'Keskustelu yhteenvedosta', 'se': u'Summeringsdiskussion' }
        self.namespaces[100] = { 'nl': u'De Wikiaan', 'se': u'Projekt' }
        self.namespaces[101] = { 'nl': u'Overleg De Wikiaan', 'se': u'Projektdiskussion' }

    def scriptpath(self, code):
        return '/w'

    def version(self, code):
        return u'1.20wmf9'

