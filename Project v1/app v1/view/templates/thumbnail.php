<?php
$html = "";
foreach($rows as $row){
    $html .= "<div id=\"".$row["id"]."\" class=\"col-sm-4 col-lg-4 col-md-4 response\">"
                ."<div class=\"thumbnail\">"
                    ."<div class=\"caption\">"
                        ."<h4><a href=\"/medico/".$row["id"]."\">".$row["nome"]."</a></h4>"
                        ."<p>CRM: ".$row["crm"]."</p>"
                        ."<p>Estado: ".$row["estado"]."</p>"
                    ."</div>"
                    ."<div class=\"ratings\">"
                        ."<p class=\"pull-right\">".numAvaliacoes($row["avaliacoes"])."</p>"
                        ."<p>"
                            .stars($row["rating"])
                        ."</p>"
                    ."</div>"
                ."</div>"
            . "</div>";
}

return $html;