
DISEASE_INFO = {

    # ==========================================
    # APPLE DISEASES
    # ==========================================

    "apple::Alternaria leaf blotch": {
        "type": "disease",

        "about": (
            "Alternaria leaf blotch is a fungal disease "
            "that affects apple leaves and can also cause "
            "spots on fruit."
        ),

        "symptoms": [
            "Irregular reddish-brown spots develop on leaves.",
            "Spots may have dark purple or blackish borders.",
            "Affected spots may enlarge and merge together.",
            "Severely affected leaves may turn yellow and fall prematurely.",
            "Small brown spots may also develop on fruit."
        ],

        "causes": [
            "Caused by fungi belonging to the Alternaria group.",
            "Warm and humid conditions can favour disease development.",
            "High rainfall and prolonged leaf wetness can encourage infection."
        ],

        "precautions": [
            "Remove fallen leaves and infected plant debris.",
            "Maintain good air circulation through appropriate pruning.",
            "Monitor leaves regularly for developing spots.",
            "Avoid unnecessary wetting of foliage.",
            "Seek agricultural advice before applying disease treatments."
        ],

        "healthy_characteristics": []
    },

    "apple::Apple scab": {
        "type": "disease",

        "about": (
            "Apple scab is a fungal disease caused by "
            "Venturia inaequalis. It affects apple leaves "
            "and fruits, particularly during cool and wet weather."
        ),

        "symptoms": [
            "Olive-green, velvety spots on leaves.",
            "Spots gradually become dark brown or black.",
            "Severely affected leaves may turn yellow and fall early.",
            "Infected fruits develop dark, rough and corky spots.",
            "Severely affected fruits may become cracked or distorted."
        ],

        "causes": [
            "Caused by the fungus Venturia inaequalis.",
            "The fungus survives in infected fallen leaves.",
            "Spores spread through wind and rain.",
            "Prolonged leaf wetness encourages infection."
        ],

        "precautions": [
            "Collect and remove infected fallen leaves.",
            "Prune trees appropriately to improve air circulation.",
            "Avoid unnecessarily wetting the foliage.",
            "Consider apple varieties known to resist apple scab.",
            "Consult a local agricultural expert for disease management."
        ],

        "healthy_characteristics": []
    },

    "apple::Black rot": {
        "type": "disease",

        "about": (
            "Apple black rot is a fungal disease caused by "
            "Botryosphaeria obtusa. It can affect leaves, "
            "fruits and branches."
        ),

        "symptoms": [
            "Leaves develop round spots with tan centres and purple borders.",
            "Fruit develops brown spots that may form concentric rings.",
            "Infected fruits may shrivel and become mummified.",
            "Branches may develop sunken or cracked cankers.",
            "Leaves on severely affected branches may wilt and turn brown."
        ],

        "causes": [
            "Caused by the fungus Botryosphaeria obtusa.",
            "The fungus can survive in infected branches and mummified fruits.",
            "Tree wounds and stressed plant tissues can favour infection."
        ],

        "precautions": [
            "Remove mummified fruits from trees.",
            "Remove dead or infected branches appropriately.",
            "Maintain good orchard sanitation.",
            "Prune trees to remove affected woody tissues.",
            "Maintain general tree health and monitor for new symptoms."
        ],

        "healthy_characteristics": []
    },

    "apple::Cedar apple rust": {
        "type": "disease",

        "about": (
            "Cedar apple rust is a fungal disease that affects "
            "apple trees. Its life cycle involves both apple "
            "trees and certain cedar or juniper plants."
        ),

        "symptoms": [
            "Yellow spots appear on the upper surface of leaves.",
            "Spots gradually become bright orange or reddish-orange.",
            "Small dark dots may develop within the spots.",
            "Raised structures may appear on the underside of leaves.",
            "Infected fruits may develop orange or brown lesions."
        ],

        "causes": [
            "Caused by rust fungi in the Gymnosporangium group.",
            "The fungus requires both apple and certain juniper or cedar hosts.",
            "Spores released from infected alternate hosts can infect apple leaves."
        ],

        "precautions": [
            "Monitor apple leaves during the growing season.",
            "Consider resistant apple varieties where appropriate.",
            "Inspect nearby susceptible juniper plants if the disease is present.",
            "Maintain general orchard health.",
            "Consult a local agricultural expert about suitable management."
        ],

        "healthy_characteristics": []
    },

    "apple::Mosaic virus": {
        "type": "disease",

        "about": (
            "Mosaic virus is a general model classification for "
            "virus-like mosaic symptoms in apple leaves. Apple "
            "mosaic virus is one possible cause, but the model "
            "prediction alone cannot confirm the exact virus."
        ),

        "symptoms": [
            "Irregular pale yellow or light-green patches may appear on leaves.",
            "Leaves may show a mosaic pattern of different green shades.",
            "Some leaves may develop ring-shaped discolouration.",
            "Affected growth may become irregular or weakened.",
            "Symptoms can resemble nutrient deficiencies or environmental damage."
        ],

        "causes": [
            "May be associated with Apple mosaic virus.",
            "Plant viruses can spread through infected propagation material.",
            "Grafting with infected plant material can transmit certain apple viruses.",
            "Other conditions can produce similar leaf patterns."
        ],

        "precautions": [
            "Use healthy, certified planting and propagation material.",
            "Avoid taking grafting material from plants suspected of infection.",
            "Have suspected infections confirmed by a plant-health expert.",
            "Remove confirmed infected planting material according to local agricultural guidance.",
            "Monitor nearby plants for similar symptoms."
        ],

        "healthy_characteristics": []
    },

    # ==========================================
    # HEALTHY APPLE
    # ==========================================

    "apple::Healthy": {
        "type": "healthy",

        "about": (
            "The model classified this apple leaf as healthy. "
            "No specific disease was identified in this prediction."
        ),

        "symptoms": [],

        "causes": [],

        "precautions": [],

        "healthy_characteristics": [
            "Leaves generally have a consistent green appearance.",
            "Leaf surfaces are generally free from unusual spots.",
            "No obvious signs of fungal growth or lesions.",
            "Leaves generally maintain their normal shape.",
            "No obvious signs of severe yellowing or damage."
        ]
    }

}


def get_disease_info(crop, disease):
    key = f"{crop}::{disease}"

    return DISEASE_INFO.get(key, {
        "type": "unknown",
        "about": "Detailed information is not yet available for this condition.",
        "symptoms": [],
        "causes": [],
        "precautions": [],
        "healthy_characteristics": []
    })
