<?php

namespace library\dom\structures;
use library\dom\object;
use library\dom\structures\components;
use library\dom\elements\components\input;
use library\dom\elements\components\inline;


class inputGroup extends components{
    public function __construct($class = 'input-group'){
        parent::__construct("div");
        $this->root->addClass($class);
        $this->root->attr("role","group");
    }
 
    public function size($class){
        $this->root->addClass($class);
    }
 
    public function create($name,$text, $spanPosition = "left", $inputType = "text"){
        assert(!isset($this->components["input"]));
        if($inputType == 'text'){
            return $this->text($name,$text,$spanPosition);
        }
        
    }
    
    private function text($name,$text,$position){
        $input = new input($name,"text");
        $input->addClass('form-control');
        $list = array();
        if(is_array($text)){
            assert(count($text) == 2);
            $span1 = $this->span($text[0]);
            $span2 = $this->span($text[1]);
            $list = array($span1,$input,$span2);
        } else {
            $span = $this->span($text);
            $span->setId("span");
            $input->attr('aria-describedby',$span->getId());
            $list = array($span,$input);
            if($position == 'right'){
                $list = array_reverse($list);
            } 
            $this->components["input"] = $input;
            $this->components["span"] = $span;
        }
        foreach($list as $item){
            $this->root->addComponent($item);
        }
        return $input;
    }
    private function span($text){
        $span = new inline("span",$text);
        $span->addClass('input-group-addon');
        return $span;
    }
    public function createCheckBox(){
        assert(!isset($this->components["input"]));
    }
    public function save(){
        return $this->root;
    }
 
}