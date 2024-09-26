const role = ["bard", "fighter", "cleric"];
const aligns = ["neutral", "good", "evil"];
const stats = {
    "fighter":{
        "hp":25,
        "atkmod":5,
        "own_ac":17,
        "dmg_dice":2,
        "dmg_die":4,
        "img":"confusedgoblin.png"
    },
    "sorcerer":{
        "hp":18,
        "atkmod":3,
        "own_ac":12,
        "dmg_dice":4,
        "dmg_die":8,
        "img":"confusedgoblin.png"
    },
    "bard":{
        "hp":21,
        "atkmod":4,
        "own_ac":15,
        "dmg_dice":4,
        "dmg_die":4,
        "img":"confusedgoblin.png"
    },
    "druid":{
        "hp":30,
        "atkmod":5,
        "own_ac":10,
        "dmg_dice":2,
        "dmg_die":4,
        "img":"confusedgoblin.png"
    },
    "goblin":{
        "hp":10,
        "atkmod":6,
        "own_ac":12,
        "dmg_dice":2,
        "dmg_die":4,
        "img":"confusedgoblin.png"
    },
    "troll":{
        "hp":40,
        "atkmod":6,
        "own_ac":8,
        "dmg_dice":4,
        "dmg_die":4,
        "img":"confusedgoblin.png"
    },
    "pickles":{
        "hp":40,
        "atkmod":20,
        "own_ac":18,
        "dmg_dice":6,
        "dmg_die":8,
        "img":"confusedgoblin.png"
    },
    "zippy":{
        "hp":5,
        "atkmod":5,
        "own_ac":8,
        "dmg_dice":5,
        "dmg_die":20,
        "img":"confusedgoblin.png"
    }

} // use this when comfortable: https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON

console.log(stats["fighter"]["atkmod"]);

btn = document.getElementById("btn");
mainbody = document.querySelector("h1");
var rolenum = Math.floor(Math.random()*3);
var alignnum = Math.floor(Math.random()*3);
var dmg = 0;
var hp = 0;
var own_ac = 0;
var enemy_hp = 0;
var enemy_dmg = 0;
var counter = 1;
var atkmod = 0;
var atk_dice = 0;
var atk_die = 0;
var enemy_ac = 0;
var enemy_atkdice = 0;
var enemy_atkdie = 0;
var enemy = [];
var character = [];

function dmg_roll(numdice, typedice){
    var dmg_total = 0;
    for (let i = 0; i < numdice; i++){
        dmg_total += Math.ceil(Math.random()*typedice);
        console.log(dmg_total);   
    }
    return dmg_total;
}
function runattack(numcheck){
    if (numcheck > enemy_ac){
        dmg = dmg_roll(atk_dice, atk_die);
        enemy_hp = enemy_hp - dmg;
        console.log("hit");
    }
}
function rundefence(numcheck){
    if (numcheck > own_ac){
        enemy_dmg = dmg_roll(enemy_atkdice, enemy_atkdie);
        hp = hp - enemy_dmg;
        console.log("ow");
    }
}


btn.addEventListener('click', function(){
    if(counter === 1){
        character = document.getElementById("character-select").value;
        enemy = document.getElementById("enemy-selecta").value;
        hp = stats[character]["hp"];
        atkmod = stats[character]["atkmod"];
        own_ac = stats[character]["own_ac"];
        atk_dice = stats[character]["dmg_dice"];
        atk_die = stats[character]["dmg_die"];
        enemy_hp = stats[enemy]["hp"];
        enemy_ac = stats[enemy]["own_ac"];
        enemy_atkmod = stats[enemy]["atkmod"];
        enemy_atkdice = stats[enemy]["dmg_dice"];
        enemy_atkdie = stats[character]["dmg_die"];
        document.getElementById("character-select").remove();
        document.getElementById("enemy-select").remove();
        document.getElementById("buttons").remove();
        document.getElementById("stats_table").remove();
        counter = counter+1;
    }
    if(counter != 3){
        var img = stats[enemy]["img"];
        console.log(rolenum, alignnum, img);
        //document.getElementById("mainbody").textContent = [role[rolenum], aligns[alignnum]];
        document.getElementById("img").src = img;
        runattack(Math.floor(Math.random()*20)+atkmod);
        rundefence(Math.floor(Math.random()*20)+enemy_atkmod);
       
        document.getElementById("btn").textContent = "Hit Again";
        document.getElementById("battletext").textContent = "You ("+character+") fought with a "+ enemy 
        +". You hit them for "+dmg+" damage.";
        document.getElementById("battletext2").textContent = "They hit you for "+enemy_dmg+" damage.";
        document.getElementById("hp").textContent = "HP = "+hp;
        document.getElementById("enemy_hp").textContent = "Enemy HP = "+enemy_hp;
        var prevhp = hp+enemy_dmg;
       if(enemy_hp <= 0){
        document.getElementById("battletext").textContent = "You ("+character+") hit a "+enemy+" for "+dmg+" damage and killed them. You win!";
        document.getElementById("battletext2").textContent = "";
        //document.getElementById("btn").remove();
        document.getElementById("btn").onclick = null;
        counter = 3;
       }
       if(enemy_hp > 0 && hp < 0){
        document.getElementById("battletext").textContent = "You ("+character+") hit the "+enemy+" for " +dmg+" then they hit you for "+enemy_dmg+" damage and you died. You lose.";
        document.getElementById("battletext2").textContent = "";
        //document.getElementById("btn").remove();
        document.getElementById("btn").onclick = null;
        counter =3;
       }
    }


})






