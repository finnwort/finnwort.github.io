const role = ["bard", "fighter", "cleric"];
const aligns = ["neutral", "good", "evil"];

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
        atkmod = +document.getElementById("atkmod").value+0;
        hp = +document.getElementById("hp").value+0;
        own_ac = +document.getElementById("own_ac").value+0;
        atk_dice = +document.getElementById("wpn_dmg_dice").value+0;
        atk_die = +document.getElementById("wpn_dmg_die").value+0;
        enemy_hp = +document.getElementById("enemy_hp").value+0;
        enemy_ac = +document.getElementById("enemy_ac").value+0;
        enemy_atkdice = +document.getElementById("enemy_atkdice").value+0;
        enemy_atkdie = +document.getElementById("enemy_atkdie").value+0;
        document.getElementById("buttons").remove();
        document.getElementById("enemybuttons").remove();
        counter = counter+1;
    }
    var img = ["image.png", "fighter.png", "cleric.png"][rolenum];
    console.log(rolenum, alignnum, img);
    //document.getElementById("mainbody").textContent = [role[rolenum], aligns[alignnum]];
    document.getElementById("img").src = img;
    runattack(Math.floor(Math.random()*20));
    rundefence(Math.floor(Math.random()*20));
    document.getElementById("urstats").textContent = "HP = "+hp;
    document.getElementById("enstats").textContent = "Enemy HP = "+enemy_hp;
    document.getElementById("btn").textContent = "Hit Again";
    document.getElementById("battletext").textContent = "You fought with a "+ role[rolenum] 
    +". You hit them for "+dmg+" damage.";
    document.getElementById("battletext2").textContent = "They hit you for "+enemy_dmg+" damage.";
    var prevhp = hp+enemy_dmg;
   if(enemy_hp < 0){
    document.getElementById("battletext").textContent = "You hit them for "+dmg+" damage and killed them. You win!";
    document.getElementById("battletext2").textContent = "";
    document.getElementById("urstats").textContent = "HP = "+prevhp;
    document.getElementById("enstats").textContent = "Enemy HP = "+enemy_hp;
    document.getElementById("btn").remove();
   }
   if(enemy_hp > 0 && hp < 0){
    document.getElementById("battletext").textContent = "You hit them for " +dmg+" then they hit you for "+enemy_dmg+" damage and you died. You lose.";
    document.getElementById("battletext2").textContent = "";
    document.getElementById("btn").remove();
   }

})






