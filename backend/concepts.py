"""
Concept lexicon: turns the words of a prompt into structured search intent.

Without an LLM, relevance depends on knowing what each word *means* for a film:
  - genre names are hard requirements ("sci-fi" -> the film must be Science Fiction)
  - moods prefer some genres and avoid others ("dark", "feel-good")
  - themes map to TMDB keyword tags, which are matched by exact name ("heist")
  - synonyms are looked for in plot overviews when a film has no matching tag

analyze_concepts() returns:
  genre_groups  list of genre lists; a relevant film has >=1 genre from every group
  soft_genres   {genre: weight} preferred genres
  avoid_genres  {genre: weight} genres that make a film a worse fit
  tags          exact TMDB keyword names to search for
  core_tags     the subset that defines a theme ("haunted house"), vs. supporting tags
  text_terms    words that signal the theme in an overview
  covered       prompt tokens explained by a concept (not part of a title)
  concepts      names of the matched concepts (for debugging / pitches)
"""
import re

# ── Genre words: naming a genre makes it a requirement ──
GENRE_WORDS = [
    (r'rom[- ]?coms?|romantic comed(?:y|ies)', ['Romance', 'Comedy']),
    (r'sci[- ]?fi|scifi|science[- ]fiction', ['Science Fiction']),
    (r'horrors?', ['Horror']),
    (r'comed(?:y|ies)|comedic', ['Comedy']),
    (r'thrillers?', ['Thriller']),
    (r'westerns?', ['Western']),
    (r'war', ['War']),
    (r'romances?|romantic|love stor(?:y|ies)', ['Romance']),
    (r'animated|animations?|anime|cartoons?', ['Animation']),
    (r'documentar(?:y|ies)|docs?', ['Documentary']),
    (r'musicals?', ['Music']),
    (r'crime', ['Crime']),
    (r'myster(?:y|ies)|whodunn?its?', ['Mystery']),
    (r'fantasy|fantasies', ['Fantasy']),
    (r'action', ['Action']),
    (r'adventures?', ['Adventure']),
    (r'dramas?|dramatic', ['Drama']),
    (r'family|kids|children\'?s', ['Family']),
    (r'histor(?:y|ical)|period pieces?|period dramas?', ['History']),
]

# ── Concepts: moods and themes ──
# hard: genre groups this concept requires (any-of within a group)
# soft / avoid: genre preferences
# tags: TMDB keyword names (matched exactly); text: overview signals
CONCEPTS = {
    # moods
    'scary': dict(pattern=r'scary|terrifying|frightening|spooky|creepy|chilling|horrifying|nightmarish',
                  hard=[['Horror', 'Thriller']], soft={'Horror': 3.0}, avoid={'Family': 1.0, 'Animation': 0.5},
                  tags=['supernatural', 'psychological horror', 'haunting'],
                  text=['terror', 'haunt', 'evil', 'nightmare', 'sinister', 'terrifying']),
    'funny': dict(pattern=r'funny|hilarious|comedic|laugh(?:s|ing)?|goofy|silly',
                  hard=[['Comedy']], text=['comedy', 'hilarious', 'comic']),
    'dark': dict(pattern=r'dark|bleak|grim|gritty|nihilistic|brutal|disturbing|twisted',
                 soft={'Thriller': 1.0, 'Crime': 1.0, 'Drama': 0.7, 'Horror': 0.7, 'Mystery': 0.7},
                 avoid={'Family': 1.5, 'Animation': 1.0, 'Comedy': 0.6, 'Romance': 0.4, 'Music': 0.5},
                 tags=['neo-noir', 'nihilism', 'dark', 'bleak', 'moral ambiguity'],
                 text=['dark', 'brutal', 'grim', 'corrupt', 'violent', 'twisted', 'murder']),
    'feelgood': dict(pattern=r'feel[- ]?good|uplifting|wholesome|heart-?warming|cheerful|happy|light-?hearted|joyful|comforting|comfort',
                     soft={'Comedy': 1.0, 'Family': 1.0, 'Romance': 0.7, 'Animation': 0.7, 'Music': 0.6, 'Adventure': 0.5},
                     avoid={'Horror': 2.0, 'War': 1.5, 'Crime': 1.0, 'Thriller': 1.0},
                     tags=['feel-good', 'heartwarming', 'friendship', 'uplifting'],
                     text=['heartwarming', 'friendship', 'joy', 'charming', 'uplifting']),
    'sad': dict(pattern=r'sad|tear-?jerkers?|heartbreaking|melanchol(?:y|ic)|depressing|emotional|cry',
                soft={'Drama': 1.5, 'Romance': 0.7}, avoid={'Comedy': 0.6, 'Action': 0.6},
                tags=['tearjerker', 'grief', 'loss of loved one', 'melancholy', 'terminal illness'],
                text=['grief', 'loss', 'dying', 'heartbreak', 'mourning']),
    'tense': dict(pattern=r'tense|suspenseful|suspense|nail-?biting|edge of (?:my|your) seat|intense|gripping',
                  soft={'Thriller': 1.5, 'Mystery': 0.7, 'Crime': 0.7, 'Horror': 0.5},
                  avoid={'Family': 1.0, 'Comedy': 0.5},
                  tags=['suspense', 'tension', 'cat and mouse'],
                  text=['race against time', 'hunt', 'trapped', 'deadly']),
    'mindbending': dict(pattern=r'mind[- ]?bending|mind[- ]?blowing|mind[- ]?boggling|trippy|surreal|cerebral|twisty|psychedelic|reality[- ]bending',
                        soft={'Science Fiction': 1.0, 'Mystery': 1.0, 'Thriller': 0.8, 'Drama': 0.4},
                        avoid={'Family': 1.0},
                        tags=['mind-bending', 'surreal', 'twist ending', 'nonlinear timeline', 'dream', 'alternate reality', 'reality vs fantasy', 'mindfuck'],
                        text=['reality', 'dream', 'memory', 'mind', 'perception', 'illusion']),
    'psychological': dict(pattern=r'psychological|psycho',
                          soft={'Thriller': 1.0, 'Drama': 0.7, 'Mystery': 0.7, 'Horror': 0.5},
                          tags=['psychological thriller', 'psychological horror', 'paranoia', 'obsession', 'mental illness'],
                          text=['obsession', 'paranoia', 'madness', 'psyche', 'unravel']),
    'slowburn': dict(pattern=r'slow[- ]?burn(?:ing)?|slow[- ]paced|atmospheric|meditative|contemplative',
                     soft={'Drama': 1.0, 'Thriller': 0.8, 'Mystery': 0.6, 'Horror': 0.5},
                     avoid={'Action': 0.6, 'Family': 0.6},
                     tags=['slow burn', 'atmospheric', 'minimalism'],
                     text=['slowly', 'quiet', 'isolated']),
    'cozy': dict(pattern=r'cozy|cosy|comfy|relaxing|chill',
                 soft={'Comedy': 1.0, 'Romance': 0.7, 'Family': 0.7, 'Animation': 0.6, 'Mystery': 0.6},
                 avoid={'Horror': 2.0, 'War': 1.5, 'Thriller': 0.6},
                 tags=['small town', 'cozy mystery', 'friendship', 'village'],
                 text=['small town', 'village', 'charming', 'quaint']),
    'weird': dict(pattern=r'weird|bizarre|strange|absurd(?:ist)?|quirky|offbeat|wacky|unhinged',
                  soft={'Comedy': 0.7, 'Fantasy': 0.7, 'Horror': 0.5, 'Science Fiction': 0.5, 'Drama': 0.3},
                  tags=['surreal', 'absurdism', 'cult film', 'quirky', 'bizarre', 'avant-garde'],
                  text=['bizarre', 'strange', 'surreal', 'absurd', 'eccentric']),
    'epic': dict(pattern=r'epic|sweeping|grand',
                 soft={'Adventure': 1.0, 'War': 0.7, 'History': 0.7, 'Fantasy': 0.7, 'Action': 0.6, 'Drama': 0.5},
                 tags=['epic', 'battle'], text=['empire', 'kingdom', 'battle', 'journey', 'saga']),
    'actionpacked': dict(pattern=r'action[- ]packed|adrenaline|explosive|high[- ]octane|thrilling',
                         hard=[['Action', 'Thriller', 'Adventure']], soft={'Action': 2.0},
                         tags=['shootout', 'car chase', 'one man army'], text=['chase', 'fight', 'explosive']),
    'indie': dict(pattern=r'indie|independent|arthouse|art[- ]house|underrated|hidden gems?|obscure|niche',
                  soft={'Drama': 0.8}, tags=['independent film', 'arthouse'],
                  text=[], prefer_obscure=True),
    'violent': dict(pattern=r'violent|bloody|brutal',
                    soft={'Action': 0.7, 'Crime': 0.7, 'Thriller': 0.7, 'Horror': 0.7, 'War': 0.5},
                    avoid={'Family': 2.0, 'Animation': 1.0},
                    tags=['violence', 'gore', 'brutality'], text=['brutal', 'bloody', 'massacre', 'violent']),

    # themes
    'timetravel': dict(pattern=r'time[- ]travel(?:l?ing)?|time loops?|time machines?|time[- ]travell?ers?',
                       soft={'Science Fiction': 1.5}, tags=['time travel', 'time loop', 'time machine', 'time paradox'],
                       text=['time travel', 'time loop', 'back in time', 'the past', 'the future']),
    'heist': dict(pattern=r'heists?|robber(?:y|ies)|capers?|bank jobs?|con artists?|cons?',
                  soft={'Crime': 1.5, 'Thriller': 0.7, 'Action': 0.5},
                  tags=['heist', 'bank robbery', 'robbery', 'con artist', 'thief', 'caper'],
                  text=['heist', 'robbery', 'steal', 'thieves', 'vault', 'con artist']),
    'zombie': dict(pattern=r'zombies?|undead', soft={'Horror': 1.5},
                   tags=['zombie', 'zombie apocalypse', 'undead'], text=['zombie', 'undead', 'infected']),
    'vampire': dict(pattern=r'vampires?|dracula', soft={'Horror': 1.2, 'Fantasy': 0.5},
                    tags=['vampire', 'dracula'], text=['vampire', 'dracula', 'blood']),
    'werewolf': dict(pattern=r'werewol(?:f|ves)|lycanthrop\w*', soft={'Horror': 1.2},
                     tags=['werewolf'], text=['werewolf']),
    'haunted': dict(pattern=r'haunted(?: houses?)?|ghosts?|hauntings?|poltergeists?|paranormal|spirits?',
                    soft={'Horror': 1.5, 'Mystery': 0.5, 'Thriller': 0.5},
                    tags=['haunted house', 'ghost', 'haunting', 'paranormal', 'poltergeist', 'supernatural'],
                    core=['haunted house', 'ghost', 'haunting', 'poltergeist'],
                    text=['haunted', 'ghost', 'spirit', 'paranormal', 'haunting']),
    'possession': dict(pattern=r'possession|possessed|exorcisms?|demons?|demonic',
                       soft={'Horror': 1.5}, tags=['demonic possession', 'exorcism', 'demon', 'possession'],
                       text=['possessed', 'exorcism', 'demon']),
    'witch': dict(pattern=r'witch(?:es|craft)?|occult|cults?|satanic',
                  soft={'Horror': 1.0, 'Fantasy': 0.5}, tags=['witch', 'witchcraft', 'cult', 'occult', 'satanism'],
                  text=['witch', 'cult', 'ritual', 'occult']),
    'slasher': dict(pattern=r'slashers?', soft={'Horror': 2.0},
                    tags=['slasher', 'masked killer', 'serial killer'], text=['killer', 'masked']),
    'serialkiller': dict(pattern=r'serial killers?|psychopaths?|murderers?',
                         soft={'Thriller': 1.0, 'Crime': 1.0, 'Horror': 0.5},
                         tags=['serial killer', 'psychopath', 'murder investigation'],
                         text=['serial killer', 'killer', 'murders']),
    'alien': dict(pattern=r'aliens?|extraterrestrials?|ufos?|alien invasions?|first contact',
                  soft={'Science Fiction': 1.5}, tags=['alien', 'alien invasion', 'extraterrestrial', 'first contact', 'ufo'],
                  text=['alien', 'extraterrestrial', 'invasion', 'ufo']),
    'space': dict(pattern=r'space|outer space|astronauts?|spaceships?|galax(?:y|ies)|interstellar|planets?',
                  soft={'Science Fiction': 2.0, 'Adventure': 0.5},
                  tags=['outer space', 'space travel', 'spaceship', 'astronaut', 'space station', 'space opera', 'space mission'],
                  text=['space', 'astronaut', 'planet', 'spaceship', 'galaxy', 'orbit']),
    'ai': dict(pattern=r'artificial intelligence|a\.?i\.?|robots?|androids?|cyborgs?|machines? learning|sentient',
               soft={'Science Fiction': 2.0}, tags=['artificial intelligence (a.i.)', 'artificial intelligence', 'robot', 'android', 'cyborg', 'humanoid robot'],
               text=['robot', 'artificial intelligence', 'android', 'cyborg', 'sentient', 'machine']),
    'dystopia': dict(pattern=r'dystopi(?:a|an|as)|totalitarian|authoritarian|big brother|orwellian',
                     soft={'Science Fiction': 1.5, 'Drama': 0.4, 'Thriller': 0.4},
                     tags=['dystopia', 'totalitarian regime', 'dystopian future', 'surveillance', 'police state'],
                     text=['dystopian', 'totalitarian', 'regime', 'oppressive', 'rebellion']),
    'apocalypse': dict(pattern=r'post[- ]?apocalyptic|apocalyp(?:se|tic)|end of the world|doomsday|nuclear war',
                       soft={'Science Fiction': 1.2, 'Action': 0.5, 'Horror': 0.4, 'Drama': 0.3},
                       tags=['post-apocalyptic future', 'apocalypse', 'end of the world', 'nuclear war', 'survival'],
                       core=['post-apocalyptic future', 'apocalypse', 'end of the world', 'nuclear war'],
                       text=['apocalypse', 'survivors', 'wasteland', 'end of the world']),
    'cyberpunk': dict(pattern=r'cyberpunk|cyber|neon[- ]?noir|hackers?',
                      soft={'Science Fiction': 2.0, 'Action': 0.5, 'Thriller': 0.5},
                      tags=['cyberpunk', 'hacker', 'virtual reality', 'dystopia'], core=['cyberpunk'], text=['hacker', 'cyber', 'virtual']),
    'superhero': dict(pattern=r'superheroe?s?|super ?powers?|marvel|dc comics|comic books?',
                      soft={'Action': 1.0, 'Science Fiction': 0.7, 'Adventure': 0.7, 'Fantasy': 0.4},
                      tags=['superhero', 'based on comic', 'super power', 'superhero team'],
                      text=['superhero', 'powers', 'villain', 'hero']),
    'revenge': dict(pattern=r'revenge|vengeance|vengeful|avenge',
                    soft={'Thriller': 1.0, 'Action': 0.7, 'Crime': 0.7, 'Drama': 0.4},
                    tags=['revenge', 'vengeance', 'rape and revenge'], text=['revenge', 'vengeance', 'avenge']),
    'comingofage': dict(pattern=r'coming[- ]of[- ]age|teen(?:age(?:rs?)?|s)?|high[- ]school|adolescen\w*|growing up|youth',
                        soft={'Drama': 1.0, 'Comedy': 0.7, 'Romance': 0.4},
                        tags=['coming of age', 'teenager', 'high school', 'adolescence', 'teen movie', 'first love'],
                        core=['coming of age', 'teenager', 'adolescence', 'high school'],
                        text=['teenager', 'high school', 'growing up', 'summer', 'adolescent']),
    'christmas': dict(pattern=r'christmas|xmas|holiday season|santa',
                      soft={'Comedy': 0.7, 'Family': 0.7, 'Romance': 0.5},
                      tags=['christmas', 'christmas party', 'santa claus', 'holiday'],
                      core=['christmas', 'christmas party', 'santa claus'],
                      text=['christmas', 'santa', 'holiday']),
    'halloween': dict(pattern=r'halloween', soft={'Horror': 1.0, 'Family': 0.3},
                      tags=['halloween'], text=['halloween']),
    'sports': dict(generic=r'sports?|athletes?|underdogs?|olympics?', pattern=r'sports?|football|soccer|boxing|boxers?|basketball|baseball|hockey|tennis|wrestling|racing|olympics?|athletes?|underdogs?',
                   soft={'Drama': 1.0, 'Comedy': 0.4},
                   tags=['sport', 'sports', 'boxing', 'basketball', 'baseball', 'american football', 'football (soccer)', 'soccer', 'ice hockey', 'tennis', 'wrestling', 'car race', 'olympic games', 'underdog', 'sports team', 'coach'],
                   text=['team', 'coach', 'championship', 'boxer', 'athlete', 'match', 'season']),
    'martialarts': dict(generic=r'martial arts?|kung[- ]?fu|swordplay', pattern=r'martial arts?|kung[- ]?fu|karate|wuxia|samurai|ninjas?|swordplay',
                        soft={'Action': 1.5, 'Adventure': 0.5},
                        tags=['martial arts', 'kung fu', 'samurai', 'wuxia', 'ninja', 'swordplay', 'karate'],
                        text=['martial', 'kung fu', 'samurai', 'sword', 'master']),
    'spy': dict(pattern=r'spy|spies|espionage|secret agents?|cia|kgb|mi6',
                soft={'Thriller': 1.0, 'Action': 1.0},
                tags=['spy', 'espionage', 'secret agent', 'cia', 'cold war', 'double agent'],
                core=['spy', 'espionage', 'secret agent', 'double agent'],
                text=['spy', 'agent', 'espionage', 'intelligence']),
    'mafia': dict(generic=r'mafia|mob(?:sters?)?|gangsters?|organi[sz]ed crime', pattern=r'mafia|mob(?:sters?)?|gangsters?|cartels?|organi[sz]ed crime|yakuza|triads?',
                  soft={'Crime': 2.0, 'Drama': 0.5, 'Thriller': 0.5},
                  tags=['mafia', 'gangster', 'organized crime', 'drug cartel', 'yakuza', 'mob boss'],
                  text=['mafia', 'gangster', 'mob', 'cartel', 'crime family']),
    'detective': dict(pattern=r'detectives?|investigations?|investigators?|sleuths?|private eyes?|cops?|police procedurals?',
                      soft={'Mystery': 1.0, 'Crime': 1.0, 'Thriller': 0.7},
                      tags=['detective', 'investigation', 'police detective', 'private detective', 'murder investigation'],
                      text=['detective', 'investigat', 'case', 'clues']),
    'noir': dict(pattern=r'neo[- ]?noir|film noir|noir',
                 soft={'Crime': 1.5, 'Thriller': 1.0, 'Mystery': 1.0, 'Drama': 0.4},
                 tags=['neo-noir', 'film noir', 'femme fatale', 'private detective'],
                 text=['detective', 'femme fatale', 'corrupt']),
    'courtroom': dict(pattern=r'courtroom|legal dramas?|lawyers?|trials?|attorneys?',
                      soft={'Drama': 1.5, 'Crime': 0.7, 'Thriller': 0.5},
                      tags=['courtroom', 'lawyer', 'trial', 'courtroom drama', 'legal drama'],
                      text=['lawyer', 'trial', 'court', 'attorney', 'jury']),
    'prison': dict(pattern=r'prisons?|jail|inmates?|prison break|escapes?',
                   soft={'Drama': 1.0, 'Crime': 0.8, 'Thriller': 0.5},
                   tags=['prison', 'prison escape', 'prisoner', 'escape'], text=['prison', 'inmate', 'escape']),
    'survival': dict(pattern=r'survival|survivors?|stranded|wilderness|castaways?',
                     soft={'Thriller': 1.0, 'Adventure': 1.0, 'Drama': 0.6},
                     tags=['survival', 'wilderness', 'stranded', 'island', 'plane crash'], core=['survival', 'stranded'],
                     text=['survive', 'stranded', 'wilderness', 'survival']),
    'disaster': dict(generic=r'disasters?|natural disasters?', pattern=r'disasters?|earthquakes?|tsunamis?|volcano(?:es)?|natural disasters?',
                     soft={'Action': 1.0, 'Thriller': 0.8, 'Drama': 0.5},
                     tags=['disaster', 'natural disaster', 'earthquake', 'tsunami', 'disaster movie'],
                     text=['disaster', 'earthquake', 'tsunami', 'storm']),
    'monster': dict(generic=r'monsters?|creatures?|creature features?|giant monsters?', pattern=r'monsters?|creatures?|kaiju|giant monsters?|creature features?|sharks?|dinosaurs?',
                    soft={'Horror': 1.0, 'Science Fiction': 0.8, 'Adventure': 0.6, 'Action': 0.5},
                    tags=['monster', 'creature', 'giant monster', 'kaiju', 'shark', 'dinosaur', 'creature feature'],
                    core=['monster', 'giant monster', 'kaiju', 'creature feature'],
                    text=['monster', 'creature', 'beast', 'shark', 'dinosaur']),
    'war': dict(generic=r'battlefields?|soldiers?|military', pattern=r'world war (?:i|ii|1|2|one|two)|ww(?:i|ii|1|2)|vietnam|battlefields?|soldiers?|military',
                soft={'War': 2.0, 'History': 0.7, 'Drama': 0.5},
                tags=['world war ii', 'world war i', 'vietnam war', 'soldier', 'battlefield', 'military'],
                text=['war', 'soldier', 'battle', 'army']),
    'roadtrip': dict(pattern=r'road[- ]?trips?|road movies?|on the road',
                     soft={'Comedy': 0.7, 'Drama': 0.7, 'Adventure': 0.7},
                     tags=['road trip', 'road movie'], text=['road trip', 'journey', 'across the country']),
    'truestory': dict(pattern=r'true stor(?:y|ies)|based on (?:a )?true|biopics?|biograph(?:y|ical)|real[- ]life',
                      soft={'Drama': 1.0, 'History': 1.0},
                      tags=['based on true story', 'biography', 'based on a true story', 'historical figure'],
                      text=['true story', 'real-life', 'life of']),
    'music': dict(generic=r'musicians?|bands?|concerts?|singers?', pattern=r'musicians?|bands?|rock(?:\'n\'| and )roll|concerts?|singers?|hip[- ]hop|jazz',
                  soft={'Music': 2.0, 'Drama': 0.6},
                  tags=['musician', 'rock band', 'concert', 'singer', 'jazz', 'hip-hop', 'music industry'],
                  text=['band', 'musician', 'singer', 'music']),
    'dance': dict(pattern=r'danc(?:e|es|ing|ers?)|ballet', soft={'Music': 1.0, 'Romance': 0.5, 'Drama': 0.5},
                  tags=['dance', 'dancer', 'ballet', 'dancing'], text=['dance', 'dancer', 'ballet']),
    'romcom': dict(pattern=r'first love|love triangle|wedding|weddings', soft={'Romance': 1.5, 'Comedy': 0.7},
                   tags=['love triangle', 'wedding', 'first love'], text=['wedding', 'love', 'romance']),
    'conspiracy': dict(pattern=r'conspirac(?:y|ies)|paranoid|cover[- ]?ups?|whistle-?blowers?',
                       soft={'Thriller': 1.5, 'Mystery': 0.7, 'Drama': 0.4},
                       tags=['conspiracy', 'paranoia', 'cover-up', 'whistleblower', 'government conspiracy'],
                       text=['conspiracy', 'cover-up', 'secret']),
    'foundfootage': dict(pattern=r'found[- ]footage|mockumentar(?:y|ies)', soft={'Horror': 1.5},
                         tags=['found footage', 'mockumentary'], text=['footage', 'camera']),
    'bodyhorror': dict(pattern=r'body[- ]horror', hard=[['Horror', 'Science Fiction']], soft={'Horror': 2.0},
                       tags=['body horror', 'mutation', 'transformation'], text=['mutation', 'body', 'flesh']),
    'gore': dict(pattern=r'gore|gory|splatter|gruesome', soft={'Horror': 2.0},
                 tags=['gore', 'splatter', 'extreme gore'], text=['gore', 'bloody', 'gruesome']),
    'erotic': dict(pattern=r'erotic(?:a)?|steamy|sensual|sexy|sex|sexual|nudity|nude',
                   soft={'Romance': 1.0, 'Drama': 1.0, 'Thriller': 0.7},
                   tags=['erotic thriller', 'erotica', 'sexuality', 'nudity', 'female nudity', 'erotic movie', 'sex'],
                   core=['erotic thriller', 'erotica', 'erotic movie'],
                   text=['seduc', 'affair', 'desire', 'erotic', 'passion', 'sexual']),
    'twist': dict(pattern=r'plot twists?|twist endings?|twists?', soft={'Thriller': 1.0, 'Mystery': 1.0},
                  tags=['twist ending', 'plot twist', 'unreliable narrator'], text=['twist', 'secret', 'truth']),
}

# Moods describe tone, not subject: their tags are only supporting evidence.
MOOD_CONCEPTS = {'scary', 'funny', 'dark', 'feelgood', 'sad', 'tense', 'mindbending', 'psychological',
                 'slowburn', 'cozy', 'weird', 'epic', 'actionpacked', 'indie', 'violent', 'twist'}

# Words that carry no intent on their own (shared with title-detection).
FILLER = {
    'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'about', 'like', 'some',
    'something', 'anything', 'movie', 'movies', 'film', 'films', 'flick', 'flicks', 'cinema', 'watch',
    'watching', 'me', 'my', 'i', 'want', 'need', 'give', 'show', 'recommend', 'recommendations', 'good',
    'great', 'best', 'top', 'is', 'it', 'that', 'this', 'very', 'really', 'kinda', 'kind', 'sort', 'are',
    'please', 'any', 'find', 'set', 'from', 'by', 'than', 'more', 'most', 'which', 'who', 'where', 'tonight',
    'today', 'night', 'weekend', 'all', 'time', 'ever', 'pick', 'picks', 'suggest', 'suggestions', 'new',
    'old', 'classic', 'classics', 'recent', 'modern', 'latest', 'rated', 'acclaimed', 'critically',
    'masterpiece', 'masterpieces', 'award', 'winning', 'oscar', 'before', 'after', 'during', 'between',
    'since', 'until', 'decade', 'era', 'century', 'short', 'long', 'quick', 'under', 'minutes', 'mins',
    'hours', 'hour', 'vibe', 'vibes', 'mood', 'feel', 'feeling', 'type', 'style', 'genre', 'genres',
    'directed', 'starring', 'featuring', 'stars', 'acted', 'helmed', 'played', 'actor', 'actress',
    'director', 'filmography', 'works', 'collection', "'s",
}


def _singular(word):
    return word[:-1] if word.endswith('s') and not word.endswith('ss') and len(word) > 3 else word


def _core_tags(spec, phrase):
    """
    Theme-defining tags for a matched phrase. A specific trigger inside a broad
    concept ("boxing" in sports, "dinosaur" in monsters) defines the theme itself,
    so its own tag is core rather than every sport or monster.
    """
    generic = spec.get('generic')
    if not generic or re.fullmatch(generic, phrase):
        return spec.get('core', spec.get('tags', []))
    word = ' '.join(_singular(w) for w in phrase.split())
    stem = word[:5]
    related = [t for t in spec.get('tags', []) if stem in t]
    return [word] + [t for t in related if t != word]


def _norm(text):
    return re.sub(r'\s+', ' ', (text or '').lower()).strip()


def analyze_concepts(text, excluded_genres=None):
    """Maps a (negation-free) prompt to structured intent. See module docstring."""
    t = _norm(text)
    excluded = set(excluded_genres or [])
    groups, soft, avoid, tags, core, terms, names = [], {}, {}, [], [], [], []
    covered = set()
    spans = []

    def claim(match):
        spans.append(match.span())
        covered.update(re.findall(r"[a-z0-9][a-z0-9'.\-]*", match.group(0)))

    def free(match):
        a, b = match.span()
        return all(b <= s or a >= e for s, e in spans)

    # Multi-word concepts first so "romantic comedies" isn't read as two genres twice
    for name, spec in sorted(CONCEPTS.items(), key=lambda kv: -len(kv[1]['pattern'])):
        for m in re.finditer(r'\b(?:' + spec['pattern'] + r')\b', t):
            if not free(m):
                continue
            claim(m)
            names.append(name)
            for group in spec.get('hard', []):
                group = [g for g in group if g not in excluded]
                if group:
                    groups.append(group)
            for g, w in spec.get('soft', {}).items():
                if g not in excluded:
                    soft[g] = soft.get(g, 0) + w
            for g, w in spec.get('avoid', {}).items():
                avoid[g] = max(avoid.get(g, 0), w)
            tags.extend(spec.get('tags', []))
            if name not in MOOD_CONCEPTS:
                core.extend(_core_tags(spec, m.group(0)))
            terms.extend(spec.get('text', []))
            break

    for pattern, genres in GENRE_WORDS:
        for m in re.finditer(r'\b(?:' + pattern + r')\b', t):
            if not free(m):
                continue
            claim(m)
            for g in genres:
                if g not in excluded:
                    groups.append([g])
                    soft[g] = soft.get(g, 0) + 1.0
            break

    # A genre the user names is never "avoided" by a mood ("dark comedy")
    required = {g for group in groups for g in group}
    avoid = {g: w for g, w in avoid.items() if g not in required and g not in soft}

    # Collapse duplicate groups, keeping order
    unique_groups = []
    for group in groups:
        if group not in unique_groups:
            unique_groups.append(group)

    return {
        'genre_groups': unique_groups,
        'soft_genres': soft,
        'avoid_genres': avoid,
        'tags': list(dict.fromkeys(tags)),
        'core_tags': list(dict.fromkeys(core)),
        'text_terms': list(dict.fromkeys(terms)),
        'covered': covered,
        'concepts': names,
        'prefer_obscure': any(CONCEPTS[n].get('prefer_obscure') for n in names),
    }
