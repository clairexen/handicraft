#!/usr/bin/env python3

import random
import re
import sys

class Wordle:
    words="""
abaci aback abaft abase abash abate abbey abbot abeam abets abhor abide
abler abode abort about above abuse abuts abuzz abyss ached aches achoo
acids acing acmes acorn acres acrid acted actor acute adage adapt added
adder addle adept adieu adman admen admit adobe adopt adore adorn adult
adzes aegis aeons aerie affix afire afoot afoul after again agape agate
agave agent agile aging agism aglow agony agree ahead aided aides ailed
aimed aired aisle alarm album alder alert algae alias alibi alien align
alike aline alive allay alley allot allow alloy aloes aloft aloha alone
along aloof aloud alpha altar alter altho altos alums amass amaze amber
amble ameba ameer amend amigo amino amirs amiss amity among amour ample
amply ampul amuck amuse angel anger angle angry angst anime anion anise
ankhs ankle annex annoy annul anode anons anted antes antic antis anvil
aorta apace apart aphid aping appal apple apply apron apses apter aptly
aquae aquas arbor arced ardor areas arena argon argot argue arias arise
armed armor aroma arose array arrow arson artsy ascot ashed ashen ashes
aside asked askew aspen aspic assay asses asset aster astir atlas atoll
atoms atone atria attar attic audio audit auger aught augur aunts aurae
aural auras autos avail avast avers avert avian avoid avows await awake
award aware awash awful awing awoke axial axing axiom axles axons azure
baaed babel babes backs bacon badge badly bagel baggy bails baits baize
baked baker bakes balds baled bales balks balky balls balms balmy balsa
banal bands bandy banes bangs banjo banks banns barbs bards bared barer
bares barfs barge barks barns baron basal based baser bases basic basil
basin basis basks bassi basso baste batch bated bates bathe baths batik
baton batty bauds bawdy bawls bayed bayou beach beads beady beaks beams
beans beard bears beast beats beaus beaux bebop becks beech beefs beefy
beeps beers beets befit befog began begat beget begin begot begun beige
being belay belch belie belle bells belly below belts bench bends bents
beret bergs berms berry berth beryl beset besom besot bests betas bevel
bible bicep biddy bided bides bidet biers bight bigot biked biker bikes
bilge bilks bills billy bimbo binds binge bingo biped birch birds birth
bison bitch bites blabs black blade blame bland blank blare blast blaze
bleak bleat bleed bleep blend blent bless blest blimp blind bling blink
blips bliss blitz bloat blobs block blocs blogs blond blood bloom blots
blown blows blued bluer blues bluff blunt blurb blurs blurt blush board
boars boast boats bobby boded bodes bogey boggy bogie bogus boils bolas
boles bolls bolts bombs bonds boned boner bones boney bongo bongs bonny
bonus boobs booby booed books booms boons boors boost booth boots booty
booze boozy borax bored borer bores borne boron bosom bossy bosun botch
bough bound bouts bowed bowel bower bowls boxed boxer boxes bozos brace
bract brads brags braid brain brake brand brash brass brats brave bravo
brawl brawn brays bread break breed brews briar bribe brick bride brief
brier brigs brims brine bring brink briny brisk broad broil broke brood
brook broom broth brown brows bruin brunt brush brusk brute bucks buddy
budge buffs buggy bugle build built bulbs bulge bulgy bulks bulky bulls
bully bumps bumpy bunch bungs bunks bunny bunts buoys burgs burka burly
burns burnt burps burro burrs burst busby bused buses bushy busts butch
butte butts buxom buyer bylaw bytes byway cabal cabby cabin cable cacao
cache cacti caddy cadet cadge cadre caged cages cagey cairn caked cakes
calfs calif calks calls calms calve calyx camel cameo camps campy canal
candy caned canes canny canoe canon canto cants caped caper capes capon
carat carbs cards cared cares caret cargo carol carom carpi carps carry
carts carve cased cases casks caste casts catch cater catty caulk cause
caved caves cavil cawed cease cedar ceded cedes celli cello cells cents
chafe chaff chain chair chalk champ chant chaos chaps chapt charm chars
chart chary chase chasm chats cheap cheat check cheek cheep cheer chefs
chess chest chews chewy chick chide chief child chile chili chill chime
chimp china chink chino chins chips chirp chits chive chock choir choke
chomp chops chord chore chose chows chuck chugs chump chums chunk churl
churn chute cider cigar cilia cinch circa cited cites civet civic civil
clack claim clamp clams clang clank clans claps clash clasp class claws
clean clear cleat clefs cleft clerk clews click cliff climb clime cling
clink clips clipt clits cloak clock clods clogs clomp clone clops close
cloth clots cloud clout clove clown cloys clubs cluck clued clues clump
clung clunk coach coals coast coats cobra cocci cocks cocky cocoa codas
coded codes codex coeds coifs coils coins coked cokes colas colds colic
colon color colts comas combo combs comer comes comet comfy comic comma
conch condo cones conga conic conks cooed cooks cooky cools coons coops
coots coped copes copra copse coral cords cored cores corks corms corns
corny corps costs cotes couch cough could count coupe coups court coven
cover coves covet covey cowed cower cowls coyer coyly cozen crabs crack
craft crags cramp crams crane crank crape craps crash crass crate crave
crawl craws craze crazy creak cream credo creed creek creel creep crepe
crept cress crest crews cribs crick cried crier cries crime crimp crisp
croak croci crock crone crony crook croon crops cross croup crowd crown
crows crude cruel cruet crumb crush crust crypt cubed cubes cubic cubit
cuffs cuing culls cults cumin cunts cupid curbs curds cured curer cures
curie curio curls curly curry curse curst curve curvy cushy cusps cuter
cutup cycle cynic cysts czars dacha daddy dados daffy daily dairy daisy
dales dally dames damns damps dance dandy dared dares darns darts dated
dates datum daubs daunt davit dawns dazed dazes deals dealt deans dears
death debar debit debts debug debut decaf decal decay decks decor decoy
decry deeds deems deeps deers defer deice deify deign deism deity delay
delis dells delta delve demon demos demur denim dense dents depot depth
derby desks deter detox deuce devil dhoti dials diary diced dices dicey
dicks dicky dicta diets digit diked dikes dills dilly dimer dimes dimly
dined diner dines dingo dings dingy dinky diode direr dirge dirks dirty
disco discs disks ditch ditto ditty divan divas dived diver dives divot
divvy dizzy djinn docks dodge dodos doers doffs doggy dogie dogma doily
doing doled doles dolls dolly dolts domed domes donor donut dooms doors
doped dopes dopey dorks dorky dorms dosed doses doted dotes dotty doubt
dough douse doves dowdy dowel downs downy dowry dowse doyen dozed dozen
dozes drabs draft drags drain drake drama drams drank drape drawl drawn
draws drays dread dream dregs dress dried drier dries drift drill drily
drink drips drive droll drone drool droop drops dross drove drown drubs
drugs druid drums drunk dryad dryer dryly ducal ducat duchy ducks ducts
duded dudes duels duets dukes dulls dully dummy dumps dumpy dunce dunes
dungs dunks dunno duped dupes dusky dusts dusty duvet dwarf dweeb dwell
dwelt dyers dying dykes eager eagle earls early earns earth eased easel
eases eaten eater eaves ebbed ebony echos edema edged edger edges edict
edify edits eerie egged egret eider eight eject eking elate elbow elder
elect elegy elfin elide elite elope elude elves email embed ember emcee
emend emery emirs emits emoji emote empty enact ended endow endue enema
enemy enjoy ennui enrol ensue enter entry enure envoy epics epoch epoxy
equal equip erase erect erode erred error erupt essay ester ether ethic
ethos euros evade evens event every evict evils evoke ewers exact exalt
exams excel execs exert exile exist exits expel expos extol extra exude
exult eying eyrie fable faced faces facet facts faded fades fagot fails
faint fairs fairy faith faked faker fakes fakir falls false famed fancy
fangs fanny farce fared fares farms farts fasts fatal fated fates fatty
fault fauna fauns favor fawns faxed faxes fazed fazes fears feast feats
fecal feces feeds feels feign feint fells felon felts femur fence fends
feral ferns ferry fests fetal fetch feted fetid fetus feuds fever fewer
fezes fiats fiber fiche fiefs field fiend fiery fifes fifth fifty fight
filch filed files filet fills filly films filmy filth final finch finds
fined finer fines finis finks finny fiord fired fires firms first firth
fishy fists fitly fiver fives fixed fixer fixes fizzy fjord flack flags
flail flair flake flaky flame flank flaps flare flash flask flats flaws
flays fleas fleck flees fleet flesh flick flied flier flies fling flint
flips flirt flits float flock floes flogs flood floor flops flora floss
flour flout flown flows flubs flues fluff fluid fluke fluky flume flung
flunk flush flute flyby flyer foals foams foamy focal focus fogey foggy
foils foist folds folio folks folly fondu fonts foods fools foots foray
force fords fores forge forgo forks forms forte forth forts forty forum
fouls found fount fours fowls foxed foxes foyer frack frail frame franc
frank frats fraud frays freak freed freer frees fresh frets friar fried
frier fries frill frisk frizz frock frogs frond front frost froth frown
froze fruit frump fryer fucks fudge fuels fugue fulls fully fumed fumes
funds fungi funks funky funny furls furor furry furze fused fuses fussy
fusty futon fuzed fuzes fuzzy gabby gable gaffe gaffs gaged gages gaily
gains gaits galas gales galls gamed gamer games gamey gamin gamma gamut
gangs gaped gapes garbs gases gasps gassy gated gates gaudy gauge gaunt
gauze gauzy gavel gawks gawky gayer gayly gazed gazer gazes gears gecko
geeks geeky geese gelds gelid genes genie genii genre gents genus geode
germs getup ghost ghoul giant gibed gibes giddy gifts gilds gills gilts
gimme gimpy gipsy girds girls girth girts gismo given gives gizmo glade
glads gland glare glass glaze gleam glean glens glide glint glitz gloat
globe globs gloom glory gloss glove glows glued glues gluey gluts glyph
gnarl gnash gnats gnawn gnaws gnome goads goals goats godly gofer going
golds golfs golly gonad goner gongs gonna goods goody gooey goofs goofy
gooks goons goose gored gores gorge gorse gotta gouge gourd gouty gowns
grabs grace grade grads graft grail grain grams grand grant grape graph
grasp grass grate grave gravy grays graze great grebe greed green greet
greys grids grief grill grime grimy grind grins gripe grips grist grits
groan groin groom grope gross group grout grove growl grown grows grubs
gruel gruff grunt guano guard guava guess guest guide guild guile guilt
guise gulag gulch gulfs gulls gully gulps gumbo gummy gunny guppy gurus
gushy gusto gusts gusty gutsy guyed gybed gybes gypsy gyros habit hacks
hafts haiku hails hairs hairy hakes haled haler hales halls halon halos
halts halve hands handy hangs hanks hanky happy hardy hared harem hares
harks harms harps harpy harry harsh harts hasps haste hasty hatch hated
hater hates hauls haunt haven haves havoc hawed hawks hayed hazed hazel
hazes heads heady heals heaps heard hears heart heath heats heave heavy
hedge heeds heels hefts hefty heirs heist helix hello helms helot helps
hence henna herbs herds heron heros hertz hewed hewer hexed hexes hicks
hided hides highs hiked hiker hikes hills hilly hilts hinds hinge hints
hippo hippy hired hires hitch hived hives hoagy hoard hoary hobby hobos
hocks hogan hoist hokey hokum holds holed holes holly homed homer homes
homey homie honed hones honey honks honor hooch hoods hooey hoofs hooks
hooky hoops hoots hoped hopes horde horns horny horse horsy hosed hoses
hosts hotel hotly hound hours house hovel hover howdy howls hubby huffs
huffy huger hulas hulks hulls human humid humor humps humus hunch hunks
hunts hurls hurry hurts husks husky hussy hutch hydra hyena hying hymen
hymns hyped hyper hypes hypos iambs icier icily icing icons ideal ideas
idiom idiot idled idler idles idols idyll idyls igloo ikons image imams
imbed imbue impel imply inane inapt inbox incur index indue inept inert
infer infix ingot inked inlay inlet inner input inset inter inure iotas
irate irked irons irony isles islet issue itchy items ivies ivory jabot
jacks jaded jades jails jambs japan japed japes jaunt jawed jazzy jeans
jeeps jeers jehad jello jells jelly jerks jerky jests jetty jewel jibed
jibes jiffy jihad jilts jimmy jinni jinns jived jives jocks johns joins
joint joist joked joker jokes jolly jolts joule joust jowls joyed judge
juice juicy julep jumbo jumps jumpy junco junks junky junta juror kabob
kapok kaput karat karma kayak kazoo kebab kebob keels keens keeps ketch
keyed khaki khans kicks kicky kiddo kiddy kills kilns kilos kilts kinda
kinds kings kinks kinky kiosk kited kites kitty kiwis klutz knack knave
knead kneed kneel knees knell knelt knife knits knobs knock knoll knots
known knows koala kooks kooky kopek krone kudos kudzu label labia labor
laced laces lacks laded laden lades ladle lager lairs laity lakes lamas
lambs lamed lamer lames lamps lance lands lanes lanky lapel lapse larch
lards large largo larks larva laser lasso lasts latch later latex lathe
laths latte lauds laugh lawns laxer laxly layer lazed lazes leach leads
leafs leafy leaks leaky leans leaps leapt learn lease leash least leave
ledge leech leeks leers leery lefts lefty legal leggy legit lemma lemme
lemon lemur lends leper letup levee level lever liars libel licit licks
liege liens lifer lifts light liked liken liker likes lilac lilts limbo
limbs limed limes limit limns limos limps lined linen liner lines lingo
links lions lipid liras lisle lisps lists liter lithe lived liven liver
lives livid llama llano loads loafs loamy loans loath lobby lobed lobes
local locks locus lodes lodge lofts lofty loges logic login logon logos
loins lolls loner longs looks looms loons loony loops loopy loose loots
loped lopes lords lorry loser loses lotto lotus louse lousy louts loved
lover loves lowed lower lowly loxes loyal luaus lubed lubes lucid lucks
lucky lucre lulls lumps lumpy lunar lunch lunge lungs lupin lupus lurch
lured lures lurid lurks lusts lusty lutes lying lymph lynch lyres lyric
macaw maced maces macho macro madam madly magic magma maids mails maims
mains maize major maker makes males malls malts mamas mambo mamma manes
manga mange mango mangy mania manic manly manna manor manse maple march
mares maria marks marry marsh marts masks mason masts match mated mates
matte matts matzo mauls mauve maven mavin maxed maxes maxim maybe mayor
mazes meals mealy means meant meats meaty mecca medal media medic meets
melds melon melts memes memos mends menus meows mercy meres merge merit
merry mesas messy metal meted meter metes metro mewed mewls miaow micra
middy midge midst miens miffs might miked mikes milch miler miles milfs
milks milky mills mimed mimes mimic mince minds mined miner mines minim
minis minks minor mints minty minus mired mires mirth misdo miser mists
misty miter mites mitts mixed mixer mixes moans moats mocha mocks modal
model modem modes mogul moire moist molar molds moldy moles molls molts
momma mommy money monks month mooch moods moody mooed moons moors moose
moots moped mopes moral moray mores morns moron mosey mossy motel motes
moths motif motor motto mound mount mourn mouse mousy mouth moved mover
moves movie mowed mower mucks mucky mucus muddy muffs mufti muggy mulch
mules mulls multi mummy mumps munch mural murks murky mused muses mushy
music musky mussy musts musty muted muter mutes mutts mynah mynas myrrh
myths nabob nacho nacre nadir naiad nails naive naked named names nanny
napes nappy narcs narks nasal nasty natal natty naval navel naves nears
neath necks needs needy neigh nerds nerdy nerve nervy nests never newel
newer newly newsy newts nexus nicer niche nicks niece nifty nigga night
nimbi nines ninja ninny ninth nippy niter nites nixed nixes noble nobly
nodal noddy nodes noels noise noisy nomad nonce nooks noose norms north
nosed noses nosey notch noted notes nouns novae novas novel noway nuder
nudes nudge nuked nukes nulls numbs nurse nutty nylon nymph oaken oakum
oared oases oasis oaten oaths obese obeys obits oboes occur ocean ocher
ochre octal octet odder oddly odium odors offal offed offer often ogled
ogles ogres oiled oinks okays okras olden older oldie olive omega omens
omits onion onset oozed oozes opals opens opera opine opium opted optic
orals orate orbit order organ osier other otter ought ounce ousts outdo
outed outer outgo ovals ovary ovens overs overt ovoid ovule owing owlet
owned owner oxbow oxide ozone paced paces packs pacts paddy padre paean
pagan paged pager pages pails pains paint pairs paled paler pales palls
palms palmy palsy panda panel panes pangs panic pansy pants panty papal
papas papaw paper parch pared pares parka parks parry parse parts party
pasha pasta paste pasts pasty patch pates paths patio patsy patty pause
paved paves pawed pawls pawns payed payee payer peace peach peaks peals
pearl pears pease pecan pecks pedal peeks peels peeps peers peeve pekoe
pelts penal pence pends penes penis penny peons peony peppy perch peril
perks perky perms pesky pesos pests petal peter petty pewee phase phial
phish phlox phone phony photo phyla piano picks picky piece piers piety
piggy pigmy piing piked piker pikes pilaf pilau pilaw piled piles pills
pilot pimps pinch pined pines pings pinks pinky pinto pints pinup pious
piped piper pipes pipit pique pitch pithy piton pivot pixel pixie pizza
place plaid plain plait plane plank plans plant plate plays plaza plead
pleas pleat plied plies plods plops plots plows ploys pluck plugs plumb
plume plump plums plunk plush poach pocks podia poems poesy poets point
poise poked poker pokes pokey polar poled poles polio polka polls polyp
ponds pones pooch poohs pools poops popes poppa poppy porch pored pores
porno ports posed poser poses posit posse posts potty pouch pound pours
pouts power poxes prank prate prawn prays preen preps press preys price
prick pricy pride pried pries prigs prime primp print prior prism privy
prize probe prods profs promo proms prone prong proof props prose prosy
proud prove prowl prows proxy prude prune psalm pshaw psych pubic pucks
pudgy puffs puffy puked pukes pulls pulps pulpy pulse pumas pumps punch
punks punts pupae pupal pupas pupil puppy puree purer purge purls purrs
purse pushy pussy putts putty pwned pygmy pylon pyres pyxes quack quads
quaff quail quake qualm quark quart quash quasi quays queen queer quell
query quest queue quick quids quiet quill quilt quips quire quirk quite
quits quoit quota quote quoth rabbi rabid raced racer races racks radar
radii radio radon rafts ragas raged rages raids rails rains rainy raise
rajah rajas raked rakes rally ramps ranch randy range rangy ranks rants
raped rapes rapid rared rarer rares rasps raspy rated rates ratio ratty
raved ravel raven raves rawer rayon razed razes razor reach react reads
ready realm reals reams reaps rearm rears rebel rebus rebut recap recta
recur redid reeds reedy reefs reeks reels reeve refer refit regal rehab
reign reins relax relay relic remit renal rends renew rents reorg repay
repel reply reran rerun reset resin rests retch retry reuse revel revue
rheas rheum rhino rhyme riced rices ricks rider rides ridge rifer riffs
rifle rifts right rigid rigor riled riles rills rimed rimes rinds rings
rinks rinse riots ripen riper risen riser rises risks risky rites ritzy
rival riven river rivet roach roads roams roans roars roast robed robes
robin robot rocks rocky rodeo roger rogue roils roles rolls roman romps
roods roofs rooks rooms roomy roost roots roped ropes roses rosin rotor
rouge rough round rouse route routs roved rover roves rowdy rowed rowel
rower royal rubes ruble ruddy ruder ruffs rugby ruing ruins ruled ruler
rules rumba rummy rumor rumps runes rungs runny runts rupee rural ruses
rusks rusts rusty saber sable sabre sacks sades sadly safer safes sagas
sager sages sahib sails saint saith salad sales sally salon salsa salts
salty salve salvo samba sames sands sandy saner sappy saree saris sassy
satay sated sates satin satyr sauce saucy sauna saved saver saves savor
savvy sawed saxes scabs scads scald scale scalp scaly scamp scams scans
scant scare scarf scars scary scats scene scent schwa scion scoff scold
scone scoop scoot scope score scorn scour scout scowl scows scram scrap
screw scrip scrod scrub scuba scuds scuff scull scums scurf seals seams
seamy sears seats sects sedan sedge seeds seedy seeks seems seeps seers
segue seize sells semen semis sends senna sense sepal sepia septa serer
serfs serge serum serve servo setup seven sever sewed sewer sexed sexes
shack shade shads shady shaft shags shahs shake shaky shale shall shalt
shame shams shank shape shard share shark sharp shave shawl sheaf shear
sheds sheen sheep sheer sheet sheik shelf shell sherd shied shies shift
shill shims shine shins shiny ships shire shirk shirr shirt shits shlep
shoal shock shoed shoes shone shook shoon shoos shoot shops shore shorn
short shots shout shove shown shows showy shred shrew shrub shrug shtik
shuck shuns shunt shush shuts shyer shyly sibyl sicks sided sides sidle
siege sieve sifts sighs sight sigma signs silks silky sills silly silos
silts since sinew singe sings sinks sinus sired siren sires sirup sisal
sises sissy sitar sited sites sixes sixth sixty sized sizer sizes skate
skeet skein skews skids skied skier skies skiff skill skimp skims skins
skips skirt skits skulk skull skunk skyed slabs slack slags slain slake
slams slang slant slaps slash slate slats slave slays sleds sleek sleep
sleet slept slews slice slick slide slier slily slime slims slimy sling
slink slips slits slobs sloes slogs sloop slope slops slosh sloth slots
slows slued slues slugs slump slums slung slunk slurp slurs slush sluts
slyer slyly smack small smart smash smear smell smelt smile smirk smite
smith smock smoke smoky smote smuts snack snafu snags snail snake snaky
snaps snare snarl sneak sneer snide sniff snipe snips snits snobs snoop
snoot snore snort snots snout snows snowy snubs snuck snuff snugs soaks
soaps soapy soars sober socks sodas sofas softy soggy soils solar soled
soles solid solos solve sonar songs sonic sonny sooth sooty soppy sorer
sores sorry sorta sorts sough souls sound soups soupy sours souse south
sowed sower space spacy spade spake spams spank spans spare spark spars
spasm spate spats spawn spays speak spear speck specs speed spell spelt
spend spent sperm spews spice spicy spied spiel spies spike spiky spill
spilt spine spins spiny spire spite spits splat splay split spoil spoke
spoof spook spool spoon spoor spore sport spots spout sprat spray spree
sprig spuds spume spunk spurn spurs spurt squab squad squat squaw squid
stabs stack staff stage stags staid stain stair stake stale stalk stall
stamp stand stank staph stare stark stars start stash state stats stave
stays stead steak steal steam steed steel steep steer stein stems stent
steps stern stews stick sties stiff stile still stilt sting stink stint
stirs stoat stock stoic stoke stole stomp stone stony stood stool stoop
stops store stork storm story stout stove stows strap straw stray strep
strew strip strop strum strut stubs stuck studs study stuff stump stung
stunk stuns stunt styes style styli suave sucks sudsy suede sugar suing
suite suits sulks sulky sully sumac sumps sunny sunup super surer surfs
surge surly sushi swabs swags swain swami swamp swank swans swaps sward
swarm swash swath swats sways swear sweat sweep sweet swell swept swift
swigs swill swims swine swing swipe swirl swish swoon swoop swops sword
swore sworn swung sylph synch syncs synod syrup tabby table taboo tabus
tacit tacks tacky tacos taffy tails taint taken taker takes tales talks
tally talon tamed tamer tames tamps tango tangs tangy tanks tansy taped
taper tapes tapir tardy tared tares taros tarot tarps tarry tarts taser
tasks taste tasty tatty taunt taupe tawny taxed taxes taxis teach teaks
teals teams tears teary tease teats techs teems teens teeny teeth telex
tells tempi tempo temps tempt tends tenet tenon tenor tense tenth tents
tepee tepid terms terns terry terse tests testy texts thank thaws thees
theft their theme there these theta thick thief thigh thine thing think
thins third thong thorn those thous three threw throb throe throw thrum
thuds thugs thumb thump thyme thymi tiara tibia ticks tidal tided tides
tiers tiffs tiger tight tikes tilde tiled tiles tills tilts timed timer
times timid tines tinge tings tinny tints tipis tipsy tired tires tiros
titan tithe title tizzy toads toady toast today toddy toffy togae togas
toils toked token tokes tolls tombs tomes tonal toned toner tones tongs
tonic tonne tools tooth toots topaz topic toque torch torsi torso torte
torts torus total toted totem totes touch tough tours touts towed towel
tower towns toxic toxin toyed trace track tract trade trail train trait
tramp trams traps trash trawl trays tread treat treed trees treks trend
tress triad trial tribe trice trick tried tries trike trill trims trios
tripe trips trite troll tromp troop trope troth trots trout troys truce
truck trued truer trues truly trump trunk truss trust truth tryst tsars
tubas tubby tubed tuber tubes tucks tufts tulip tulle tumid tummy tumor
tunas tuned tuner tunes tunic tunny turds turfs turns tusks tutor tutus
tuxes twain twang tweak tweed tweet twerk twerp twice twigs twill twine
twins twirl twist twits tying tykes typed types typos tyros tzars udder
ulcer ulnae ulnas ultra umbel umber umiak umped unbar uncle uncut under
undid undue unfit unify union unite units unity unman unpin unsay unset
untie until unwed unzip upend upped upper upset urban urged urges urine
usage users usher using usual usurp usury uteri utter uvula vacua vague
vales valet valid valor value valve vamps vanes vaped vapes vapid vapor
vases vasts vault vaunt veeps veers vegan veils veins velds veldt venal
vends venom vents venue verbs verge verse verve vests vetch vexed vexes
vials viand vibes vicar viced vices video views vigil vigor viler villa
vines vinyl viola viols viper viral vireo virus visas vised vises visit
visor vista vital vivas vivid vixen vizor vocal vodka vogue voice voids
voile voles volts vomit voted voter votes vouch vowed vowel vulva vying
wacko wacks wacky waded wader wades wadis wafer wafts waged wager wages
wagon waifs wails waist waits waive waked waken wakes waled wales walks
walls waltz wands waned wanes wanly wanna wants wards wares warms warns
warps warts warty wasps waste watch water watts waved waver waves waxed
waxen waxes weals weans wears weary weave wedge weeds weedy weeks weeps
weepy weest wefts weigh weird weirs welch welds wells welsh welts wench
wends wetly whack whale whams wharf whats wheal wheat wheel whelk whelp
whens where whets which whiff while whims whine whiny whips whirl whirr
whirs whisk whist white whits whizz whole whoop whore whorl whose wicks
widen wider widow width wield wight wikis wilds wiled wiles wills wilts
wimps wimpy wince winch winds windy wined wines wings winks winos wiped
wiper wipes wired wires wiser wises wisps wispy witch witty wives wizes
woken wolfs woman wombs women woods woody wooed wooer woofs wooly woozy
words wordy works world worms wormy worry worse worst worth would wound
woven wowed wrack wraps wrapt wrath wreak wreck wrens wrest wrier wring
wrist write writs wrong wrote wroth wrung wryer wryly xenon xylem yacht
yacks yahoo yanks yards yarns yawed yawls yawns yeahs yearn years yeast
yells yelps yeses yield yocks yodel yogin yogis yoked yokel yokes yolks
young yours youth yowls yucca yucks yucky yummy yuppy zebra zebus zeros
zests zilch zincs zings zippy zombi zonal zoned zones zooms""".split()

    def __init__(self, regex, includes, excludes):
        self.regex = re.compile(regex)
        self.includes = list(sorted(includes))
        self.excludes = list(sorted(excludes))

    def findGuesses(self, samples=10):
        self.matches = list()

        for word in self.words:
            if self.regex.fullmatch(word):
                for c in self.includes:
                    if c in word:
                        word = word[:(i := word.index(c))] + \
                               c.upper() + word[i+1:]
                    else:
                        break
                else:
                    for c in self.excludes:
                        if c in word: break
                    else:
                        self.matches.append(word)

        n = min(len(self.matches), samples)
        print(f"-- {n} of {len(self.matches)} guesses --")
        for m in sorted(random.sample(self.matches, n)):
            print(f" {m.lower()}   [{m}]")
        print()

    def findProbes(self, useMatches=False, levels=2, samples=10):
        covered = set(self.includes + self.excludes)

        bestScore = [-1 for i in range(levels)]
        matches = [None for i in range(levels)]

        words = self.words
        if useMatches:
            words = list([m.lower() for m in self.matches])

        for word in words:
            score = len(set(word) - covered)
            for lvl in range(levels):
                if score > bestScore[lvl]:
                    bestScore = bestScore[0:lvl] + [score] + bestScore[lvl:-1]
                    matches = matches[0:lvl] + [[word]] + matches[lvl:-1]
                    break
                if score == bestScore[lvl]:
                    matches[lvl].append(word)
                    break

        for lvl in range(levels):
            if matches[lvl] is None:
                continue
            n = min(len(matches[lvl]), samples)
            print(f"-- {n} of {len(matches[lvl])} {'matching ' if useMatches else ''}probes with score {bestScore[lvl]} --")
            for m in sorted(random.sample(matches[lvl], n)):
                k = "".join([c.upper() if c in covered else c for c in m])
                print(f" {m}   [{k}]")
            print()

    def findFirst():
        data = [{} for i in range(5)]
        gdata = {}

        for word in Wordle.words:
            for i, c in enumerate(word):
                if c not in data[i]:
                    data[i][c] = 0
                data[i][c] += 1
                if c not in gdata:
                    gdata[c] = 0
                gdata[c] += 1

        maxScore = 0
        results = []

        for word in Wordle.words:
            score = 0
            for i, c in enumerate(word):
                score += data[i][c]
                score += 7 * gdata[c]
            score *= len(set(word))
            if score > maxScore:
                maxScore = score
                results = []
            results.append(word)

        print(f"{len(Wordle.words)=} {len(results)=} {maxScore=}")
        print(f"Good words to start with: {', '.join(sorted(random.sample(results, 10)))}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 worldle.py regex include-chars exclude-chars")
        print("Examle: python3 wordle.py '.[^ta][^t]..' 'at' 'sonedumpxhc'")
        print()
        Wordle.findFirst()
        sys.exit(1)

    wordle = Wordle(*sys.argv[1:])
    wordle.findGuesses()
    wordle.findProbes(True)
    wordle.findProbes()