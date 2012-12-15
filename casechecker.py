#!/usr/bin/python
# -*- coding: utf-8  -*-
""" Script to enumerate all pages on the wiki and find all titles
with mixed latin and cyrilic alphabets.
"""
#
# (C) Pywikipedia bot team, 2006-2012
#
# Distributed under the terms of the MIT license.
#
__version__ = '$Id$'

#
# Permutations code was taken from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/190465
#
def xuniqueCombinations(items, n):
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xuniqueCombinations(items[i+1:], n-1):
                yield [items[i]] + cc
# End of permutation code
#

#
# Windows Concole colors
# This code makes this script Windows ONLY!!!
# Feel free to adapt it to another platform
#
# Adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/496901
#
STD_OUTPUT_HANDLE= -11

FOREGROUND_BLUE = 0x01 # text color contains blue.
FOREGROUND_GREEN= 0x02 # text color contains green.
FOREGROUND_RED  = 0x04 # text color contains red.
FOREGROUND_INTENSITY = 0x08 # text color is intensified.
BACKGROUND_BLUE = 0x10 # background color contains blue.
BACKGROUND_GREEN= 0x20 # background color contains green.
BACKGROUND_RED  = 0x40 # background color contains red.
BACKGROUND_INTENSITY = 0x80 # background color is intensified.

FOREGROUND_WHITE = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

try:
    import ctypes
    std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
except:
    std_out_handle = None


def SetColor(color):
    if std_out_handle:
        try:
            return ctypes.windll.kernel32.SetConsoleTextAttribute(
                std_out_handle, color)
        except:
            pass

    if color == FOREGROUND_BLUE: print '(b:',
    if color == FOREGROUND_GREEN: print '(g:',
    if color == FOREGROUND_RED: print '(r:',

# end of console code

import os
import sys, query, re, codecs
import wikipedia as pywikibot


class CaseChecker( object ):
    msgRename = {
        'ar': u'تغيير اسم لحالة مخلوطة',
        'en': u'mixed case rename',
        'ru': u'[[ВП:КЛ]]',
    }
    msgDeleteRedirect = {
        'en': u'This redirect contains identical looking Cyrillic and Latin letters in its title',
        'ru': u'[[ВП:КЛ]] Перенаправление содержит смесь кириллицы и латиницы в названии',
    }
    textDeleteRedirect = {
        'en': u'{{db-r3|bot=CaseChecker}}\n\nThis redirect used to point to %s',
        'ru': u'{{Db-redirtypo|[[ВП:КЛ]] Перенаправление на %s содержало смесь кириллицы и латиницы в названии}}',
    }
    msgLinkReplacement = {
        'en': u'Case Replacements',
        'ar': u'استبدالات الحالة',
        'ru': u'[[ВП:КЛ]]',
    }

    # These words are always in one language, even though they could be typed
    # in both
    alwaysInLocal = [ u'СССР', u'Как', u'как' ]
    alwaysInLatin = [ u'II', u'III' ]

    localUpperLtr = u'ЁІЇЎАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯҐ'
    localLowerLtr = u'ёіїўабвгдежзийклмнопрстуфхцчшщъыьэюяґ'
    localLtr = localUpperLtr + localLowerLtr

    localSuspects = u'АВЕКМНОРСТХІЁЇаеорсухіёї'
    latinSuspects = u'ABEKMHOPCTXIËÏaeopcyxiëï'

    # possibly try to fix one character mistypes in an alternative keyboard
    # layout
    localKeyboard = u'йцукенгшщзфывапролдячсмить'
    latinKeyboard = u'qwertyuiopasdfghjklzxcvbnm'

    romanNumChars = u'IVXLMC'
    # all letters that may be used as suffixes after roman numbers:  "Iый"
    romannumSuffixes = localLowerLtr
    romanNumSfxPtrn = re.compile(
        u'^[' + romanNumChars + ']+[' + localLowerLtr + ']+$')

    whitelists = {
        'ru': u'ВП:КЛ/Проверенные',
        }

    latLtr = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    lclClrFnt = u'<font color=green>'
    latClrFnt = u'<font color=brown>'
    suffixClr = u'</font>'

    wordBreaker = re.compile(u'[ _\-/\|#[\]():]')
    stripChars = u' \t,'

    titles = True
    links = False
    aplimit = None
    apfrom = u''
    title = None
    replace = False
    stopAfter = -1
    wikilog = None
    wikilogfile = 'wikilog.txt'
    failedTitles = 'failedTitles.txt'
    nosuggestions = 'nosuggestions.txt'
    doFailed = False
    titleList = None
    autonomous = False
    namespaces = []
    filterredir = 'nonredirects'

    def __init__(self):

        for arg in pywikibot.handleArgs():
            if arg.startswith('-from'):
                if arg.startswith('-from:'):
                    self.apfrom = arg[6:]
                else:
                    self.apfrom = pywikibot.input(u'Which page to start from: ')
            elif arg.startswith('-reqsize:'):
                self.aplimit = int(arg[9:])
            elif arg == '-links':
                self.links = True
            elif arg == '-linksonly':
                self.links = True
                self.titles = False
            elif arg == '-replace':
                self.replace = True
            elif arg == '-redir':
                self.filterredir = 'all'
            elif arg == '-redironly':
                self.filterredir = 'redirects'
            elif arg.startswith('-limit:'):
                self.stopAfter = int(arg[7:])
            elif arg == '-autonomous' or arg == '-a':
                self.autonomous = True
            elif arg.startswith('-ns:'):
                self.namespaces.append( int(arg[4:]) )
            elif arg.startswith('-wikilog:'):
                self.wikilogfile = arg[9:]
            elif arg.startswith('-failedlog:'):
                self.failedTitles = arg[11:]
            elif arg == '-failed':
                self.doFailed = True
            else:
                pywikibot.output(u'Unknown argument %s.' % arg)
                pywikibot.showHelp()
                pywikibot.stopme()
                sys.exit()

        if self.namespaces == [] and not self.doFailed:
            if self.apfrom == u'':
                # 0 should be after templates ns
                self.namespaces = [14, 10, 12, 0]
            else:
                self.namespaces = [0]

        if self.aplimit is None:
            self.aplimit = 200 if self.links else 'max'

        if not self.doFailed:
            self.queryParams = { 'action'        : 'query',
                                 'generator'     : 'allpages',
                                 'gaplimit'      : self.aplimit,
                                 'gapfilterredir': self.filterredir}
        else:
            self.queryParams = { 'action'        : 'query' }
            if self.apfrom != u'': pywikibot.output(u'Argument "-from" is ignored with "-failed"')

        propParam = 'info'
        if self.links:
            propParam += '|links|categories'
            self.queryParams['pllimit'] = 'max'
            self.queryParams['cllimit'] = 'max'

        self.queryParams['prop'] = propParam

        self.site = pywikibot.getSite()

        if len(self.localSuspects) != len(self.latinSuspects):
            raise ValueError(u'Suspects must be the same size')
        if len(self.localKeyboard) != len(self.latinKeyboard):
            raise ValueError(u'Keyboard info must be the same size')

        if not os.path.isabs(self.wikilogfile):
            self.wikilogfile = pywikibot.config.datafilepath(self.wikilogfile)
        self.wikilog = self.OpenLogFile(self.wikilogfile)

        if not os.path.isabs(self.failedTitles):
            self.failedTitles = pywikibot.config.datafilepath(self.failedTitles)

        if self.doFailed:
            with codecs.open(self.failedTitles, 'r', 'utf-8') as f:
                self.titleList = [self.Page(t) for t in f]
            self.failedTitles += '.failed'

        self.lclToLatDict = dict([(ord(self.localSuspects[i]),
                                   self.latinSuspects[i])
                                  for i in xrange(len(self.localSuspects))])
        self.latToLclDict = dict([(ord(self.latinSuspects[i]),
                                   self.localSuspects[i])
                                  for i in xrange(len(self.localSuspects))])

        if self.localKeyboard is not None:
            self.lclToLatKeybDict = dict(
                [(ord(self.localKeyboard[i]),
                  self.latinKeyboard[i])
                 for i in xrange(len(self.localKeyboard))])
            self.latToLclKeybDict = dict(
                [(ord(self.latinKeyboard[i]),
                  self.localKeyboard[i])
                 for i in xrange(len(self.localKeyboard))])
        else:
            self.lclToLatKeybDict = {}
            self.latToLclKeybDict = {}

        badPtrnStr = u'([%s][%s]|[%s][%s])' \
                     % (self.latLtr, self.localLtr, self.localLtr, self.latLtr)
        self.badWordPtrn = re.compile(u'[%s%s]*%s[%s%s]*'
                                      % (self.latLtr, self.localLtr,
                                         badPtrnStr, self.latLtr,
                                         self.localLtr))

        # Get whitelist
        self.knownWords = set()
        self.seenUnresolvedLinks = set()

        # TODO: handle "continue"
        if self.site.lang in self.whitelists:
            wlpage = self.whitelists[self.site.lang]
            pywikibot.output(u'Loading whitelist from %s' % wlpage)
            wlparams = {
                'action'    : 'query',
                'prop'      : 'links',
                'titles'    : wlpage,
                'redirects' : '',
                'indexpageids' : '',
                'pllimit'   : 'max',
            }

            data = query.GetData(wlparams)
            if len(data['query']['pageids']) == 1:
                pageid = data['query']['pageids'][0]
                links = data['query']['pages'][pageid]['links']

                allWords = [nn for n in links for nn in self.FindBadWords(n['title'])]

                self.knownWords = set(allWords)
#                kw = set()
#                for w in allWords:
#                    if len(self.ProcessTitle(w)[1]) > 0:
#                        kw.add(w)
#                self.knownWords = kw

            else:
                raise ValueError(u'The number of pageids is not 1')

            pywikibot.output(u'Loaded whitelist with %i items' % len(self.knownWords))
            if pywikibot.verbose and len(self.knownWords) > 0:
                pywikibot.output(u'Whitelist: %s' % u', '.join([self.MakeLink(i, False) for i in self.knownWords]))
        else:
            pywikibot.output(u'Whitelist is not known for language %s'
                             % self.site.lang)

    def RunQuery(self, params):
        while True:
            # Get data
            data = query.GetData(params)

            # Process received data
            yield data

            # Clear any continuations first
            if 'clcontinue' in params: del params['clcontinue']
            if 'plcontinue' in params: del params['plcontinue']

            if 'query-continue' not in data:
                if 'gapcontinue' in params: del params['gapcontinue']
                break

            qc = data['query-continue']
            # First continue properties only, once done, continue with allpages
            if 'categories' in qc or 'links' in qc:
                if 'categories' in qc: params.update(qc['categories'])
                if 'links' in qc: params.update(qc['links'])
            elif 'allpages' in qc:
                params.update(qc['allpages'])
            else:
                raise ValueError(u'Unexpected query-continue values: %s' % qc)
            continue

    def Run(self):
        try:
            self.lastLetter = ''

            if not self.doFailed:
                for namespace in self.namespaces:
                    self.currentTitle = None
                    self.queryParams['gapnamespace'] = namespace
                    self.queryParams['gapfrom'] = self.apfrom
                    for data in self.RunQuery(self.queryParams):
                        self.ProcessDataBlock(data)
            else:
                self.currentTitle = None
                batchSize = 10
                for batchStart in xrange(0, len(self.titleList), batchSize):
                    self.queryParams['titles'] = self.titleList[batchStart:batchStart+batchSize]
                    for data in self.RunQuery(self.queryParams):
                        self.ProcessDataBlock(data)

            print "*" * 29, "Done"

        except:
            pywikibot.output(u'Exception at Title = %s, Next = %s' % (self.currentTitle, self.apfrom))
            try:
                import traceback
                pywikibot.output(traceback.format_exc())
            except:
                pywikibot.output(u'Unable to print exception info')
            raise

    def ProcessDataBlock(self, data):
        if 'query' not in data or 'pages' not in data['query']:
            return

        firstItem = True
        for pageID, page in data['query']['pages'].iteritems():

            printed = False
            title = page['title']
            self.currentTitle = title

            if 'missing' in page:
                continue

            if firstItem:
                if self.lastLetter != title[0]:
                    pywikibot.ui.output('Processing %s\n' % title)
                    self.lastLetter = title[0]
                firstItem = False

            if self.titles:
                err = self.ProcessTitle(title)
                if err:
                    changed = False
                    if self.replace:
                        if len(err[1]) == 1:
                            newTitle = err[1][0]
                            # choice = pywikibot.inputChoice(u'Move %s to %s?' % (title, newTitle), ['Yes', 'No'], ['y', 'n'])
                            editSummary = pywikibot.translate(self.site, self.msgRename)
                            dst = self.Page(newTitle)

                            if 'redirect' in page:
                                src = self.Page(title)
                                redir = src.getRedirectTarget()
                                redirTitle = redir.title(asLink = True, textlink = True)

                                if not dst.exists():
                                    src.move(newTitle, editSummary, movesubpages=True)
                                    changed = True

                                replErrors = False
                                for p in src.getReferences(follow_redirects = False):
                                    if p.namespace() == 2:
                                        continue
                                    oldText = p.get(get_redirect = True)
                                    newText = self.ReplaceLink(oldText, title, newTitle)
                                    if not self.PutNewPage(p, newText, [self.MakeMoveSummary(title, newTitle)]):
                                        replErrors = True
                                if not replErrors:
                                    editSummary = pywikibot.translate(self.site, self.msgDeleteRedirect)
                                    newText = pywikibot.translate(self.site, self.textDeleteRedirect) % redirTitle
                                    src.put(newText, editSummary, minorEdit=False)
                                    changed = True

                            elif not dst.exists():
                                src = self.Page(title)
                                if page['ns'] == 14:
                                    import category
                                    dst = self.Page(newTitle)
                                    bot = category.CategoryMoveRobot(
                                        src.title(withNamespace=False),
                                        dst.title(withNamespace=False),
                                        self.autonomous,
                                        editSummary + u' ' + self.MakeMoveSummary(title, newTitle),
                                        True)
                                    bot.run()
                                else:
                                    src.move(newTitle, editSummary, movesubpages=True)
                                changed = True

                    if not changed:
                        if len(err[1]) > 0:
                            self.AppendLineToLog(self.failedTitles, title)
                        else:
                            self.AddNoSuggestionTitle(title)

                        self.WikiLog(u"* " + err[0])
                        printed = True

            if self.links:
                allLinks = None
                if 'links' in page:
                    allLinks = page['links']
                if 'categories' in page:
                    if allLinks:
                        allLinks = allLinks + page['categories']
                    else:
                        allLinks = page['categories']

                if allLinks:
                    pageObj = None
                    pageTxt = None
                    msg = []
                    foundSuggestions = False

                    for l in allLinks:
                        ltxt = l['title']
                        err = self.ProcessTitle(ltxt)
                        if err:
                            if len(err[1]) > 0:
                                foundSuggestions = True
                            elif self.AddNoSuggestionTitle(ltxt):
                                continue

                            newTitle = None
                            if self.replace:
                                newTitle = self.PickTarget(title, ltxt, err[1])
                                if newTitle:
                                    if pageObj is None:
                                        pageObj = self.Page(title)
                                        pageTxt = pageObj.get()

                                    msg.append(self.MakeMoveSummary(ltxt, newTitle))

                                    pageTxt = self.ReplaceLink(pageTxt, ltxt, newTitle)

                            if not newTitle:
                                if not printed:
                                    self.WikiLog(u"* %s: link to %s" % (self.MakeLink(title, False), err[0]))
                                    printed = True
                                else:
                                    self.WikiLog(u"** link to %s" % err[0])


                    if pageObj is not None:
                        if self.PutNewPage(pageObj, pageTxt, msg):
                                # done, no need to log anything
                                foundSuggestions = False

                    if foundSuggestions:
                        self.AppendLineToLog(self.failedTitles, title)


            if self.stopAfter > 0:
                self.stopAfter -= 1
                if self.stopAfter == 0:
                    raise ValueError(u'Stopping because we are done')

    def WikiLog(self, text):
        pywikibot.output(text)
        self.wikilog.write(text + u'\n')
        self.wikilog.flush()

    def FindBadWords(self, title):
        for m in self.badWordPtrn.finditer(title):
            yield title[m.span()[0] : m.span()[1]]

    def ProcessTitle(self, title):

        badWords = list(self.FindBadWords(title))

        if len(badWords) > 0:
            # Allow known words, allow any roman numerals with local suffixes
            badWords = set([i for i in badWords if i not in self.knownWords and self.romanNumSfxPtrn.match(i) is not None])

        if len(badWords) == 0 or self.Page(title).isImage():
            return None

        count = 0
        ambigBadWords = set()
        ambigBadWordsCount = 0
        mapLcl = {}
        mapLat = {}

        for badWord in badWords:

            # See if it would make sense to treat the whole word as either
            # cyrilic or latin
            mightBeLat = mightBeLcl = True
            for l in badWord:
                if l in self.localLtr:
                    if mightBeLat and l not in self.localSuspects:
                        mightBeLat = False
                else:
                    if mightBeLcl and l not in self.latinSuspects:
                        mightBeLcl = False
                    if l not in self.latLtr: raise ValueError(u'Assert failed')

            # Some words are well known and frequently mixed-typed
            if mightBeLcl and mightBeLat:
                if badWord in self.alwaysInLocal:
                    mightBeLat = False
                elif badWord in self.alwaysInLatin:
                    mightBeLoc = False

            if mightBeLcl:
                mapLcl[badWord] = badWord.translate(self.latToLclDict)
            if mightBeLat:
                mapLat[badWord] = badWord.translate(self.lclToLatDict)
            if mightBeLcl and mightBeLat:
                ambigBadWords.add(badWord)
                # Cannot do len(ambigBadWords) because they might be duplicates
                ambigBadWordsCount += 1
            if not mightBeLcl and not mightBeLat:
                # try to match one of the knownWords
                bwLen = len(badWord)
                kw = [w for w in self.knownWords if len(w) == bwLen]
                for p in xrange(bwLen):
                    if len(kw) == 0:
                        break
                    c = badWord[p]
                    co = ord(c)
                    if co in self.latToLclDict:
                        c2 = self.latToLclDict[co]
                    elif co in self.lclToLatDict:
                        c2 = self.lclToLatDict[co]
                    else:
                        c2 = None
                    kw = [w for w in kw if p < len(w) and (w[p] == c or (c2 is not None and w[p] == c2))]
                if len(kw) > 1:
                    pywikibot.output(u"Word '%s' could be treated as more than one known words" % badWord)
                elif len(kw) == 1:
                    mapLcl[badWord] = kw[0]
            count += 1

        infoText = self.MakeLink(title)
        possibleAlternatives = []

        if len(mapLcl) + len(mapLat) - ambigBadWordsCount < count:
            # We cannot auto-translate - offer a list of suggested words
            suggestions = mapLcl.values() + mapLat.values()
            if len(suggestions) > 0:
                infoText += u", word suggestions: " + u', '.join(
                    [self.ColorCodeWord(t) for t in suggestions])
            else:
                infoText += u", no suggestions"
        else:

            # Replace all unambiguous bad words
            for k,v in mapLat.items() + mapLcl.items():
                if k not in ambigBadWords:
                    title = title.replace(k,v)

            if len(ambigBadWords) == 0:
                # There are no ambiguity, we can safelly convert
                possibleAlternatives.append(title)
                infoText += u", convert to " + self.MakeLink(title)
            else:
                # Try to pick 0, 1, 2, ..., len(ambiguous words) unique
                # combinations from the bad words list, and convert just the
                # picked words to cyrilic, whereas making all other words as
                # latin character.
                for itemCntToPick in xrange(0, len(ambigBadWords)+1):
                    title2 = title
                    for uc in xuniqueCombinations(list(ambigBadWords),
                                                  itemCntToPick):
                        wordsToLat = ambigBadWords.copy()
                        for bw in uc:
                            title2 = title2.replace(bw, mapLcl[bw])
                            wordsToLat.remove(bw)
                        for bw in wordsToLat:
                            title2 = title2.replace(bw, mapLat[bw])
                        possibleAlternatives.append(title2)

                if len(possibleAlternatives) > 0:
                    infoText += u", can be converted to " + u', '.join(
                        [self.MakeLink(t) for t in possibleAlternatives])
                else:
                    infoText += u", no suggestions"

        return (infoText, possibleAlternatives)

    def PickTarget(self, title, original, candidates):

        if len(candidates) == 0:
            return None

        if len(candidates) == 1:
            return candidates[0]

        pagesDontExist = []
        pagesRedir = {}
        pagesExist = []

        for newTitle in candidates:
            dst = self.Page(newTitle)
            if not dst.exists():
                pagesDontExist.append(newTitle)
            elif dst.isRedirectPage():
                pagesRedir[newTitle] = dst.getRedirectTarget().title()
            else:
                pagesExist.append(newTitle)

        if len(pagesExist) == 1:
            return pagesExist[0]
        elif len(pagesExist) == 0 and len(pagesRedir) > 0:
            if len(pagesRedir) == 1:
                return pagesRedir.keys()[0]
            t = None
            for k,v in pagesRedir.iteritems():
                if not t:
                    t = v # first item
                elif t != v:
                    break
            else:
                # all redirects point to the same target
                # pick the first one, doesn't matter what it is
                return pagesRedir.keys()[0]

        if not self.autonomous:
            pywikibot.output(u'Could not auto-decide for page %s. Which link should be chosen?' % self.MakeLink(title, False))
            pywikibot.output(u'Original title: ', newline=False)
            self.ColorCodeWord(original + "\n", True)
            count = 1
            for t in candidates:
                if t in pagesDontExist: msg = u'missing'
                elif t in pagesRedir: msg = u'Redirect to ' + pagesRedir[t]
                else: msg = u'page exists'
                self.ColorCodeWord(u'  %d: %s (%s)\n'
                                    % (count, t, msg), True)
                count += 1

            answers = [str(i) for i in xrange(0, count)]
            choice = int(pywikibot.inputChoice(
                u'Which link to choose? (0 to skip)',
                answers, [a[0] for a in answers]))
            if choice > 0:
                return candidates[choice-1]

        return None

    def ColorCodeWord(self, word, toScreen = False):

        if not toScreen: res = u"<b>"
        lastIsCyr = word[0] in self.localLtr
        if lastIsCyr:
            if toScreen: SetColor(FOREGROUND_GREEN)
            else: res += self.lclClrFnt
        else:
            if toScreen: SetColor(FOREGROUND_RED)
            else: res += self.latClrFnt

        for l in word:
            if l in self.localLtr:
                if not lastIsCyr:
                    if toScreen: SetColor(FOREGROUND_GREEN)
                    else: res += self.suffixClr + self.lclClrFnt
                    lastIsCyr = True
            elif l in self.latLtr:
                if lastIsCyr:
                    if toScreen: SetColor(FOREGROUND_RED)
                    else: res += self.suffixClr + self.latClrFnt
                    lastIsCyr = False
            if toScreen: pywikibot.output(l, newline=False)
            else: res += l

        if toScreen: SetColor(FOREGROUND_WHITE)
        else: return res + self.suffixClr + u"</b>"

    def AddNoSuggestionTitle(self, title):
        if title in self.seenUnresolvedLinks:
            return True
        self.seenUnresolvedLinks.add(title)

        params = {
            'action'    : 'query',
            'list'      : 'backlinks',
            'bltitle'   : title,
            'bllimit'   : '50',
        }

        data = query.GetData(params)
        cl = 0
        redirs = 0
        if 'backlinks' in data['query']:
            bl = data['query']['backlinks']
            cl = len(bl)
            redirs = len([i for i in bl if 'redirect' in i])

        if cl > 0 and 'query-continue' in data:
            count = '50+'
        else:
            count = str(cl if cl > 0 else 'no backlinks')

        self.AppendLineToLog(self.nosuggestions, u'* %s (%s%s)' %
                             (self.MakeLink(title), count, u', %d redirects' % redirs if redirs > 0 else u''))
        return False

    def PutNewPage(self, pageObj, pageTxt, msg):
        title = pageObj.title(asLink = True, textlink = True)
        coloredMsg = u', '.join([self.ColorCodeWord(m) for m in msg])
        if pageObj.get(get_redirect = True) == pageTxt:
            self.WikiLog(u"* Error: Text replacement failed in %s (%s)" % (self.MakeLink(title, False), coloredMsg))
        else:
            pywikibot.output(u'Case Replacements: %s' % u', '.join(msg))
            try:
                pageObj.put(
                    pageTxt,
                    u'%s: %s'
                    % (pywikibot.translate(
                        self.site,
                        self.msgLinkReplacement),
                        u', '.join(msg)))
                return True
            except KeyboardInterrupt:
                raise
            except:
                self.WikiLog(u"* Error: Could not save updated page %s (%s)" % (self.MakeLink(title, False), coloredMsg))
        return False

    def MakeMoveSummary(self, fromTitle, toTitle):
        return u'[[%s]]→[[%s]]' % (fromTitle, toTitle)

    def MakeLink(self, title, colorcode=True):
        prf = u'' if self.Page(title).namespace() == 0 else u':'
        cc = u'|««« %s »»»' % self.ColorCodeWord(title) if colorcode else u''
        return u"[[%s%s%s]]" % (prf, title, cc)

    def OpenLogFile(self, filename):
        try:
            return codecs.open(filename, 'a', 'utf-8')
        except IOError:
            return codecs.open(filename, 'w', 'utf-8')

    def AppendLineToLog(self, filename, text):
        with self.OpenLogFile(filename) as f:
            f.write(text + u'\n')

    def Page(self, title):
        return pywikibot.Page(self.site, title)

    def ReplaceLink(self, text, oldtxt, newtxt):

        frmParts = [str.strip(self.stripChars) for str in self.wordBreaker.split(oldtxt)]
        toParts = [str.strip(self.stripChars) for str in self.wordBreaker.split(newtxt)]

        if len(frmParts) != len(toParts):
            raise ValueError(u'Splitting parts do not match counts')
        for i in xrange(0, len(frmParts)):
            if len(frmParts[i]) != len(toParts[i]):
                raise ValueError(u'Splitting parts do not match word length')
            if len(frmParts[i]) > 0:
                text = text.replace(frmParts[i][0].lower() + frmParts[i][1:], toParts[i][0].lower() + toParts[i][1:])
                text = text.replace(frmParts[i][0].upper() + frmParts[i][1:], toParts[i][0].upper() + toParts[i][1:])

        return text


if __name__ == "__main__":
    try:
        bot = CaseChecker()
        bot.Run()
    finally:
        pywikibot.stopme()
