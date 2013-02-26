# -*- coding: utf-8  -*-

__version__ = '$Id$'

import family

# The wikidata family

class Family(family.WikimediaFamily):
    def __init__(self):
        super(Family, self).__init__()
        self.name = 'wikidata'
        self.langs = {
            'wikidata': 'www.wikidata.org',
            'repo': 'wikidata-test-repo.wikimedia.de',
            'client': 'wikidata-test-client.wikimedia.de',
        }

        # Override defaults
        self.namespaces[1]['repo'] = [u'Talk', u'Item talk']
        self.namespaces[0]['repo'] = [u'', u'Item']

        # Most namespaces are inherited from family.Family.
        # Translation used on all wikis for the different namespaces.
        # (Please sort languages alphabetically)
        # You only need to enter translations that differ from _default.
        self.namespaces[4] = {
            '_default': [u'Wikidata', u'WD', 'Project'],
            'client': u'Testwiki',
            'repo': u'Testwiki',
        }
        self.namespaces[5] = {
            '_default': [u'Wikidata talk', u'WT', 'Project talk'],
            'client': u'Testwiki talk',
            'repo': u'Testwiki talk',
        }
        self.namespaces[102] = {
            'repo': u'Property',
        }
        self.namespaces[103] = {
            'repo': u'Property talk',
        }
        self.namespaces[104] = {
            'repo': u'Query',
        }
        self.namespaces[105] = {
            'repo': u'Query talk',
        }
        self.namespaces[120] = {
            'wikidata': u'Property',
        }
        self.namespaces[121] = {
            'wikidata': u'Property talk',
        }
        self.namespaces[122] = {
            'wikidata': u'Query',
        }
        self.namespaces[123] = {
            'wikidata': u'Query talk',
        }
        self.namespaces[1198] = {
            'wikidata': u'Translations',
        }
        self.namespaces[1199] = {
            'wikidata': u'Translations talk',
        }        

    def shared_data_repository(self, code, transcluded=False):
        """Always return a repository tupe. This enables testing whether
        the site opject is the repository itself, see Site.is_data_repository()

        """
        if transcluded:
            return(None, None)
        else:
            return ('wikidata',
                    'wikidata') if code == 'wikidata' else ('repo', 'wikidata')
