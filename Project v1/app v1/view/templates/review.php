<?php

$html = "<div id=\"reviewBox\" class=\"reviewbox\">"
            ."<div class=\"text-right\">"
                ."<a class=\"btn btn-success\">Avaliar</a>"
            ."</div>"
            ."<hr>";
foreach($rows as $row){
    $html .= reviewRow($row);
}
$html .= "</div>"   
         ."<div class=\"text-center\">"
                ."<button class=\"btn btn-lg btn-warning hidden loader\"><span class=\"glyphicon glyphicon-refresh glyphicon-refresh-animate\"></span> Carregando...</button>"
         ."</div>";
return $html;