# -*- coding: utf-8 -*-

__version__ = '$Id$'

# The new wikivoyage family that is hosted at wikimedia

import family

class Family(family.Family):
    def __init__(self):
        family.Family.__init__(self)
        self.name = 'wikivoyage'
        self.langs = {
            'de': u'de.wikivoyage.org',
            'en': u'en.wikivoyage.org',
            'fr': u'fr.wikivoyage.org',
            'it': u'it.wikivoyage.org',
            'nl': u'nl.wikivoyage.org',
            'ru': u'ru.wikivoyage.org',
            'sv': u'sv.wikivoyage.org',
        }

        self.namespaces[1] = self.namespaces.get(1, {})
        self.namespaces[1][u'fr'] = [u'Discuter']
        self.namespaces[2] = self.namespaces.get(2, {})
        self.namespaces[2][u'ru'] = [u'\u0423\u0447\u0430\u0441\u0442\u043d\u0438\u0446\u0430']
        self.namespaces[2][u'de'] = [u'Benutzerin']
        self.namespaces[3] = self.namespaces.get(3, {})
        self.namespaces[3][u'ru'] = [u'\u041e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435 \u0443\u0447\u0430\u0441\u0442\u043d\u0438\u0446\u044b']
        self.namespaces[3][u'fr'] = [u'Discussion Utilisateur']
        self.namespaces[3][u'de'] = [u'Benutzerin Diskussion']
        self.namespaces[4] = self.namespaces.get(4, {})
        self.namespaces[4][u'ru'] = [u'Wikivoyage', u'WV']
        self.namespaces[4][u'fr'] = [u'Wikivoyage', u'WV']
        self.namespaces[4][u'en'] = [u'Wikivoyage', u'WV']
        self.namespaces[4][u'nl'] = [u'Wikivoyage', u'WV']
        self.namespaces[4][u'de'] = [u'Wikivoyage']
        self.namespaces[4][u'sv'] = [u'Wikivoyage', u'WV']
        self.namespaces[4][u'it'] = [u'Wikivoyage', u'WV']
        self.namespaces[5] = self.namespaces.get(5, {})
        self.namespaces[5][u'ru'] = [u'\u041e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435 Wikivoyage']
        self.namespaces[5][u'fr'] = [u'Discussion Wikivoyage']
        self.namespaces[5][u'en'] = [u'Wikivoyage talk']
        self.namespaces[5][u'nl'] = [u'Overleg Wikivoyage']
        self.namespaces[5][u'de'] = [u'Wikivoyage Diskussion']
        self.namespaces[5][u'sv'] = [u'Wikivoyagediskussion']
        self.namespaces[5][u'it'] = [u'Discussioni Wikivoyage']
        self.namespaces[6] = self.namespaces.get(6, {})
        self.namespaces[6][u'ru'] = [u'Image', u'\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435']
        self.namespaces[6][u'fr'] = [u'Image']
        self.namespaces[6][u'en'] = [u'Image']
        self.namespaces[6][u'nl'] = [u'Image', u'Afbeelding']
        self.namespaces[6][u'de'] = [u'Image', u'Bild']
        self.namespaces[6][u'sv'] = [u'Image', u'Bild']
        self.namespaces[6][u'it'] = [u'Image', u'Immagine']
        self.namespaces[7] = self.namespaces.get(7, {})
        self.namespaces[7][u'ru'] = [u'Image talk', u'\u041e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f']
        self.namespaces[7][u'fr'] = [u'Image talk', u'Discussion Fichier', u'Discussion Image']
        self.namespaces[7][u'en'] = [u'Image talk']
        self.namespaces[7][u'nl'] = [u'Image talk', u'Overleg afbeelding']
        self.namespaces[7][u'de'] = [u'Image talk', u'Bild Diskussion']
        self.namespaces[7][u'sv'] = [u'Image talk', u'Bilddiskussion']
        self.namespaces[7][u'it'] = [u'Image talk', u'Discussioni immagine']
        self.namespaces[104] = self.namespaces.get(104, {})
        self.namespaces[104][u'de'] = [u'Thema']
        self.namespaces[104][u'it'] = [u'Tematica']
        self.namespaces[105] = self.namespaces.get(105, {})
        self.namespaces[105][u'de'] = [u'Thema Diskussion']
        self.namespaces[105][u'it'] = [u'Discussioni tematica']
        self.namespaces[106] = self.namespaces.get(106, {})
        self.namespaces[106][u'de'] = [u'Nachrichten', u'News']
        self.namespaces[106][u'it'] = [u'Notizie']
        self.namespaces[11] = self.namespaces.get(11, {})
        self.namespaces[11][u'fr'] = [u'Discussion Mod\xe8le']
        self.namespaces[13] = self.namespaces.get(13, {})
        self.namespaces[13][u'fr'] = [u'Discussion Aide']
        self.namespaces[13][u'sv'] = [u'Hj\xe4lp diskussion']
        self.namespaces[15] = self.namespaces.get(15, {})
        self.namespaces[15][u'fr'] = [u'Discussion Cat\xe9gorie']
        self.namespaces[107] = self.namespaces.get(107, {})
        self.namespaces[107][u'de'] = [u'Nachrichten Diskussion', u'News Diskussion']
        self.namespaces[107][u'it'] = [u'Discussioni notizie']
        self.namespaces[9] = self.namespaces.get(9, {})
        self.namespaces[9][u'sv'] = [u'MediaWiki diskussion']
        self.namespaces[100] = self.namespaces.get(100, {})
        self.namespaces[100][u'de'] = [u'Portal']
        self.namespaces[100][u'it'] = [u'Portale']
        self.namespaces[103] = self.namespaces.get(103, {})
        self.namespaces[103][u'de'] = [u'Wahl Diskussion']
        self.namespaces[103][u'it'] = [u'Discussioni elezione']
        self.namespaces[102] = self.namespaces.get(102, {})
        self.namespaces[102][u'de'] = [u'Wahl']
        self.namespaces[102][u'it'] = [u'Elezione']
        self.namespaces[101] = self.namespaces.get(101, {})
        self.namespaces[101][u'de'] = [u'Portal Diskussion']
        self.namespaces[101][u'it'] = [u'Discussioni portale']

    def scriptpath(self, code):
        return u'/w'

    def shared_image_repository(self, code):
        return ('commons', 'commons')
