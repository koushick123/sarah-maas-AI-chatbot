from tinydb import TinyDB, Query
# This script initializes a TinyDB database to store prompts and analyses related to Sarah J. Maas's works.

prompt_db = TinyDB('prompt_logs.json')

# Insert a prompt for feminist analysis of a book
prompt_db.insert({
"book_title": "Sarah Maas - Assassins Blade",
  "feminist_analysis": {
    "female_agency": [
      {
        "character": "Celaena Sardothien",
        "description": "Struggles with depression but makes an independent decision to train with Rowan and embrace her magical identity."
      }
    ],
    "gender_roles_and_expectations": [
      {
        "scene": "Celaena's rejection of Maeve's manipulations",
        "analysis": "Challenges patriarchal expectations by asserting emotional and magical autonomy."
      }
    ],
    "sisterhood_and_female_alliances": [
      {
        "relationship": "Manon and the Thirteen",
        "analysis": "Depicts complex female leadership and loyalty dynamics among witches."
      }
    ],
    "intersectionality": [
      {
        "character": "Aelin Galathynius",
        "analysis": "Navigates both political leadership and personal trauma, emphasizing layered identities."
      }
    ],
    "oppressive_structures": [
      {
        "scene": "Maeve’s control over information and power",
        "analysis": "Represents a matriarchal oppression mirroring patriarchal control mechanisms."
      }
    ]
  }
})

# Insert multiple prompts for deconstruction of binary oppositions and character identity in Sarah J. Maas's works
prompt_db.insert_multiple(
    [
        {
            "book_name": "Sarah Maas - Assassins Blade",
            "book_title": "Assassins Blade",
            "chapter": "The Stranger in the Mist",
            "type": "deconstruction",
            "tags": ["identity", "contradiction", "binary", "ambiguity"],
            "excerpt": "“Just like how you proper men surrounded a defenseless girl in an alley?”",
            "analysis_note": "This passage exposes a contradiction between the stranger's words and actions, subverting earlier meanings about her character. "
                          "The phrase 'proper men' is initially ambiguous, but Yrene's reaction suggests that it refers to those who would harm or exploit women. "
                          "However, the stranger herself wields two long daggers with dripping blood, blurring the binary opposition between victimizer and protector. "
                          "This ambiguity challenges the reader's expectations about her character, "
                          "as she simultaneously critiques the behavior of 'proper men' while also embodying a violent and potentially oppressive power dynamic."
        },
        {
            "book_name": "Sarah Maas - Throne of Glass",
            "book": "Throne of Glass",
             "type": "deconstruction",
             "tags": ["binary", "ambiguity", "identity"],
             "chapter": "Not specified",
             "excerpt": "\"That he's very attached to her. Possibly in love with her.\"",
             "analysis_note": "The passage exposes a binary opposition between the Duke's perception of Lady Lillian as 'foolish' and Kaltain's understanding of Dorian's "
                              "attachment to her as 'tragic'. This contradiction highlights the ambiguity surrounding Lady Lillian's character identity, "
                              "which is further complicated by Kaltain's own motivations. The Duke's dismissal of Dorian's feelings for Lady Lillian as 'impossible' "
                              "suggests a breakdown in traditional meaning structures, implying that societal norms and expectations may not apply to this particular situation. "
                              "This internal contradiction challenges the reader's understanding of the characters and their relationships, adding complexity to the narrative."
        },
        {
            "book_name": "Sarah Maas - Crown of Midnight",
            "book": "Crown of Midnight",
             "type": "deconstruction",
             "tags": ["identity", "contradiction", "binary", "ambiguity"],
             "chapter": "The Wagon",
             "excerpt": "\"I found a riddle, and my friends have been debating its answer for weeks. We even have a bet going about it,\" she said as vaguely as she could.",
             "analysis_note": "This passage exposes an internal contradiction in Celaena's character identity. On one hand, she presents herself as a carefree and impudent"
                              " individual who is willing to make a bet with Yellowlegs. This aligns with her earlier persona as a charming and confident assassin. "
                              "However, the ambiguity surrounding her 'friends' and their motivations suggests that Celaena may be struggling with her own identity and loyalties. "
                              "The binary opposition between her public image and private doubts creates tension and subverts the traditional meaning structure of a single, "
                              "fixed character identity."
        },
        {
            "book_name": "Sarah Maas - Heir of Fire",
            "book": "Heir of Fire",
             "type": "deconstruction",
             "tags": ["binary opposition", "ambiguity", "character identity"],
             "chapter": "Chaol's Experiment with Dorian",
             "excerpt": "I think that this kingdom could use a healer as its queen.",
             "analysis_note": "The passage exposes a binary opposition between Chaol's perception of Sorscha and her actual character. On one hand, Chaol sees Sorscha as "
                              "'truly stunning' and thinks she would make a great queen, implying a romantic interest. On the other hand, Sorscha is revealed to be sad "
                              "and unresponsive to Chaol's suggestion, subverting his earlier expectations. This ambiguity challenges the traditional notion of a character's "
                              "identity being fixed or easily readable, instead highlighting the complexity and nuance of human emotions."
        },
        {
            "book_name": "Sarah Maas - Queen of Shadows",
            "book": "Queen of Shadows",
             "type": "deconstruction",
             "tags": ["identity", "contradiction", "binary"],
             "chapter": "The Blackbeaks",
             "excerpt": "Good choice, witchling, Manon said, and the word was a challenge and an order. She turned away, but glanced over her shoulder. "
                        "Welcome to the Blackbeaks.",
             "analysis_note": "This passage exposes a binary opposition between belonging and mistake, as Elide reflects on her decision to join the Blackbeaks. "
                              "The phrase 'strange' suggests that this feeling of belonging is not entirely coherent with her earlier identity as a witchling. "
                              "This internal contradiction highlights the complexity of Elide's character development, as she navigates the blurred lines "
                              "between loyalty, duty, and personal desire. The ambiguity surrounding her decision to join the Blackbeaks subverts "
                              "traditional notions of good vs. evil, instead presenting a nuanced exploration of moral gray areas."
        },
        {
            "book_name": "Sarah Maas - Empire of Storms",
            "book": "Empire of Storms",
             "type": "deconstruction",
             "tags": ["binary opposition", "ambiguity", "character identity"],
             "chapter": "Ilium's Significance",
             "excerpt": "The Mycenians are nothing more than a myth—they were banished three hundred years ago. If you’re looking for a symbol, they’re fairly "
                        "outdated—and divisive.",
             "analysis_note": "This passage exposes a binary opposition between the perceived value of the Mycenians as a myth and their actual historical significance. "
                              "The text initially presents the Mycenians as 'nothing more than a myth', implying that they are irrelevant and outdated. "
                              "However, Aelin's subsequent explanation reveals that the Mycenians were once powerful crime lords who ruled Ilium and played a crucial role in "
                              "winning wars. This binary opposition between myth and reality subverts earlier meanings by highlighting the complexity of historical narratives. "
                              "Additionally, this passage also raises questions about character identity, as Aelin's connection to the Mycenians is ambiguous. "
                              "Her cousin's calculation of reasons why Ilium is vital suggests that Aelin's motivations may be rooted in her own sense of identity and purpose."
        },
        {
            "book_name": "Sarah Maas - Tower of Dawn",
            "book": "Tower of Dawn",
             "type": "deconstruction",
             "tags": ["binary opposition", "ambiguity", "character identity"],
             "chapter": "N/A",
             "excerpt": "\"I don't blame them for abandoning it if it's this cold in the summer,\" Nesryn muttered. \"Imagine it in winter.\"",
             "analysis_note": "The passage exposes a binary opposition between the harsh, unforgiving environment of the Fells and the perceived abandonment of the tower. "
                              "This contrast subverts earlier meanings by suggesting that even in the most inhospitable places, there can be beauty (the 'cool' air) "
                              "and value (the intact archway). Nesryn's muttering also hints at ambiguity in her own character identity, as she struggles to reconcile her "
                              "initial relief with the potential danger of the tower. This internal contradiction highlights the complexity of Nesryn's personality and "
                              "underscores the theme of uncertainty that pervades the novel."
        },
        {
            "book_name": "Sarah Maas - Kingdom of Ash",
            "book": "Kingdom of Ash",
             "chapter": "26",
             "type": "deconstruction",
             "tags": ["identity", "contradiction", "binary"],
             "excerpt": "When you finish breaking me apart for the day, how does it feel to know that you are still nothing?",
             "analysis_note": "The passage exposes a contradiction in Cairn's character identity. On one hand, he grins and says 'Some fire left in you, it seems. "
                              "Good.' suggesting a sense of admiration or even affection towards Aelin. However, this is immediately subverted by his subsequent statement "
                              "that she is the only reason he has an oath to fulfill, implying that without her, he is nothing. This binary opposition between Cairn's "
                              "perceived value and actual worth highlights the theme of identity crisis in the series. The ambiguity surrounding Cairn's character raises "
                              "questions about the nature of loyalty, power dynamics, and the blurred lines between good and evil."
        }
    ]
)

prompt_db.insert_multiple([
    {
        "book_name": "Sarah Maas - Assassins Blade",
        "book_title": "Assassins Blade",
        "feminist_analysis":
          { "female_agency":
                [
                    { "character": "Celaena Sardothien",
                      "description": "Makes an independent decision to train with Rowan and embrace her magical identity despite being surrounded by patriarchal expectations."
                      }
                ],
            "oppressive_structures":
              [
                  { "scene": "The mention of slavery and the need for women like Maeve to protect people on a list, implying a matriarchal oppression",
                    "analysis": "Represents a form of oppressive control mirroring patriarchal structures"
                    }
              ]
          }
    },
    {    "book_name": "Sarah Maas - Throne of Glass",
         "book_title": "Throne of Glass",
        "feminist_analysis":
            { "female_agency":
                  [
                      { "character": "Celaena Sardothien",
                        "description": "Asserts her autonomy by refusing Chaol's restrictions and making independent decisions about her training."
                        }
                  ],
                "gender_roles_and_expectations":
                    [
                        { "scene": "Celaena's rejection of Chaol's expectations",
                          "analysis": "Challenges patriarchal expectations by asserting emotional and magical autonomy, resisting Chaol's attempts to control her."
                          }
                    ]
            }
    },
    {
        "book_name": "Sarah Maas - Crown of Midnight",
        "book_title": "Crown of Midnight",
        "feminist_analysis":
            { "female_agency":
                  [
                      { "character": "Celaena Sardothien",
                        "description": "Makes an independent decision to recite the names of her dead and swing her pickax into the overseer's gut, "
                                       "asserting control over her own life."
                        }
                  ],
                "oppressive_structures":
                    [
                        { "scene": "The overseer's whip and Adarlan's oppressive regime",
                          "analysis": "Represents patriarchal oppression and control mechanisms."
                          }
                    ]
            }
    },
    {
        "book_name": "Sarah Maas - Heir of Fire",
        "book_title": "Heir of Fire",
        "feminist_analysis":
            { "sisterhood_and_female_alliances":
                  [
                      { "relationship": "Manon and the Thirteen",
                        "analysis": "Depicts complex female leadership and loyalty dynamics among witches."
                        }
                  ],
                "oppressive_structures":
                    [
                        { "scene": "The villagers' fear and hostility towards Manon, a powerful witch",
                          "analysis": "Represents patriarchal oppression and marginalization of women with perceived power or 'otherness'."
                          }
                    ]
            }
    },
    {
        "book_name": "Sarah Maas - Queen of Shadows",
        "book_title": "Queen of Shadows",
        "feminist_analysis":
            { "sisterhood_and_female_alliances":
                  [
                      { "relationship": "Manon and the Thirteen",
                        "analysis": "Depicts complex female leadership and loyalty dynamics among witches."
                        }
                  ],
                "oppressive_structures":
                    [
                        { "scene": "Maeve’s control over information and power",
                          "analysis": "Represents a matriarchal oppression mirroring patriarchal control mechanisms."
                          }
                    ]
            }
    },
    {
        "book_name": "Sarah Maas - Empire of Storms",
        "book_title": "Empire of Storms",
        "feminist_analysis":
            { "female_agency":
                  [
                      { "character": "Manon Blackbeak",
                        "description": "Makes an independent decision to train and assert her magical identity despite her grandmother's expectations."
                        }
                  ],
                "sisterhood_and_female_alliances":
                    [
                        { "relationship": "Iskra, Petrah, and Manon",
                          "analysis": "Depicts complex female leadership and loyalty dynamics among witches, with Iskra and Petrah intervening to protect their Matrons "
                                      "and Manon."
                          }
                    ]
            }
    },
    {
        "book_name": "Sarah Maas - Tower of Dawn",
        "book_title": "Tower of Dawn",
        "feminist_analysis":
            {
                "female_agency":
                    [
                        { "character": "Nesryn",
                          "description": "Makes an independent decision to share her family's celebration with Chaol and asserts emotional autonomy."
                          }
                    ],
                "oppressive_structures":
                    [
                        { "scene": "The stranger's instructions for women to yell 'fire' in case of attack, emphasizing the need for self-defense",
                          "analysis": "Represents a patriarchal expectation that women must take responsibility for their own safety and protection."
                          }
                    ]
            }
    },
    {
        "book_name": "Sarah Maas - Kingdom of Ash",
        "book_title": "Kingdom of Ash",
        "feminist_analysis":
            {
                "female_agency":
                    [
                        { "character": "Manon Blackbeak",
                          "description": "Exhibits agency by refusing to back down from her grandmother and the Yellowlegs Matron, despite being outnumbered."
                          }
                    ],
                "sisterhood_and_female_alliances":
                    [
                        { "relationship": "Manon's relationship with her grandmother",
                          "analysis": "Highlights tension between maternal figures and their daughters, showcasing complex female dynamics."
                          }
                    ]
            }
    }
])

# Save the database
prompt_db.close()