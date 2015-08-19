<?php

namespace library\dom\structures;
use library\dom\structures\components;
use library\dom\elements\components\block;
use library\dom\elements\components\link;
use library\dom\elements\components\img;
use library\dom\elements\components\text;


class media extends components{
    private $align;
    private $position;
    
    public function __construct($position = "left", $align = "top"){
        parent::__construct("div");
        $this->root->addClass("media");
        $this->align = $align;
        $this->position = $position;
    }
 
    public function create($imgsrc,$title,$text){
        $div = new block("div");
        $div->addClass("media-".$this->position);
        if($this->align != 'top'){
            $div->addClass("media-".$this->align);
        }
        $a = new link();
        $div->addComponent($a);
        $img = new img($imgsrc);
        $img->addClass("media-object");
        $a->addComponent($img);
        
        $body = new block("div");
        $body->addClass('media-body');
        $h4 = new block('h4');
        $h4->addClass("media-heading");
        $body->addComponent($h4);
        $highlight = new text($title);
        $h4->addComponent($highlight);
        $txt = new text($text);
        $body->addComponent($txt);
        
        if($this->position == 'left'){
            $this->root->addComponent($div);
            $this->root->addComponent($body);
        } else {
            $this->root->addComponent($body);
            $this->root->addComponent($div);
        }
        $this->components["div"][] = $div;
        $this->components["div"][] = $body;
        $this->components["img"][] = $img;
        $this->components["a"][] = $a;         
        
    }

    public function save(){
        return $this->root;
    }
 
}