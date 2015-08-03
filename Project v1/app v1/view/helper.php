<?php
function numAvaliacoes($avaliacoes){
    return ($avaliacoes > 0 ? $avaliacoes:"Sem").($avaliacoes > 1? " avaliações":" avaliação");
}
function stars($rating){
    $html="";
    for($i = 1; $i <= 5; $i++){    
        $html .= "<i class=\"fa fa-star ".($i <= $rating?"yellow":"gray")."\"></i>";
    }
    return $html;
}

function reviewRow($row){
    $html = "<div id=\"".$row["id"]."\" class=\"row response\">"
                ."<div class=\"col-md-12\">"
                    .stars($row["rating"])
                    ."<span class=\"small-margin-left\">".$row["nome"]."</span>"
                    ."<span class=\"pull-right\">".$row["data"]."</span>"
                    ."<p>".$row["comentario"]."</p>"
                ."</div>"
            ."</div>"
            . "<hr>";
    return $html;
}
function optionList($rows){
    $html="";
    foreach($rows as $row){    
        $html .= "<option value=".$row["id"].">".$row["nome"]."</option>";
    }
    return $html;
}